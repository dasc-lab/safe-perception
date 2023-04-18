# Make sure we use the system-installed Python backend
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/usr/bin/python3"

using PythonCall, GeometryBasics, LinearAlgebra, StaticArrays
using ColorTypes, MeshCat
using UUIDs
using Logging, Printf

include("utils.jl")  # Includes static matrix types, rototranslation functions

Point3f = Point3f0  # Required if using newer versions of GeometryBasics
cv = pyimport("cv2")
np = pyimport("numpy")
py = pybuiltins

function get_matches(img1::Py, img2::Py, detector_type::String="orb"; nfeatures=1000, use_flann=true)::Tuple{Py, Py, Py}
    """
    Get matches between two grayscale images.
    Args:
        img1: first image
        img2: second image 
        detector_type: "orb", "sift", or "akaze"
        nfeatures: max features to retain for all detectors except AKAZE. Does NOT guarantee this many features.

    Returns: (keypoints from image 1, keypoints from image 2, matches); all as relevant OpenCV objects
    """

    # Set up detector and distance to use for matching
    if detector_type == "orb"
        # Lower threshold = more features
        detector = cv.ORB_create(nfeatures=nfeatures, scoreType=1, fastThreshold=2)
        feature_norm = cv.NORM_HAMMING
    elseif detector_type == "sift"
        # May not work depending on version of OpenCV; patent expired 03-2020
        # Higher threshold = more features
        detector = cv.SIFT_create(nfeatures=nfeatures, edgeThreshold=100)
        feature_norm = cv.NORM_L2
    elseif detector_type == "akaze"
        # AKAZE does not directly support the nfeatures argument
        # Lower threshold = more features
        detector = cv.AKAZE_create(threshold=0.0001)
        feature_norm = cv.NORM_HAMMING
    elseif detector_type == "surf"
        # Note: SURF is patented and requires a special build flag / contrib opencv
        # TODO(rgg): explore whether adjustments to hessianThreshold are needed to increase number of features extracted.
        detector = cv.SURF_create(nfeatures=nfeatures)
        feature_norm = cv.NORM_L2
    else
        return
    end

    # Find the keypoints and descriptors.
    @info "Finding keypoints"
    @time begin
        kp1, des1 = detector.detectAndCompute(img1, py.None)
        kp2, des2 = detector.detectAndCompute(img2, py.None)
    end
    
    # See https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html for details
    # FLANN tuned for ORB
    if use_flann
        FLANN_INDEX_LSH = 6
        index_params = py.dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12 recommended by docs
                            key_size = 12,     # 20
                            multi_probe_level = 1) # 2
        search_params = py.dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        @info "Matching keypoints with FLANN"
        @time all_matches = flann.knnMatch(des1,des2,k=2)  # Get two nearest neighbors for ratio test
        matches = py.list()  # TODO(rgg): make this a Julia vector?
        # Check if matches is empty or python None
        matches_empty = pyeval(Bool, "all_matches is None or len(all_matches) == 0", Main, (all_matches=all_matches,))
        if matches_empty
            @warn "No matches found"
            return kp1, kp2, matches
        end
        # Print first few elements of all_matches
        @debug "First few elements of all_matches:"
        for i in 1:min(5, py.len(all_matches))
            @debug all_matches[i]
        end
        # Print length of matches found
        @info "Number of matches found by FLANN: " py.len(all_matches)
        # Ratio test as per Lowe's paper.
        # If the closest match is much closer than the second closest, it's a good match.
        for match_pair in all_matches
            # Print pair of matches
            @debug match_pair
            # Sometimes FLANN is unable to produce two matches
            if pyconvert(Bool, py.len(match_pair) < 2)
                continue
            else
                m = match_pair[0]
                n = match_pair[1]
            end
            if pyconvert(Bool, m.distance < 0.7*n.distance)
                matches.append(m)
            end
        end
    else
        # Create BFMatcher object. Hamming distance for ORB, L2 for SIFT/SURF.
        bf = cv.BFMatcher(feature_norm, crossCheck=true)
        # Match descriptors.
        matches = bf.match(des1,des2)
    end
    # Sort matches in the order of their distance--not needed downstream?
    #matches = py.sorted(matches, key = x -> x.distance)
    n_matches = pyconvert(Int, py.len(matches))
    if n_matches < 200
        @warn "Low matched feature count across frames: " n_matches
    end

    return kp1, kp2, matches
end

function get_point_3d(K_inv::SM3{Float32}, point::Tuple{Float32, Float32}, depth::Float32)::SV3{Float32}
    """
    Takes a point in u,v and returns a point in x, y, z using the inverse
    camera intrinsics matrix and the known depth (z) of the point.
    Slow, use for visualization only.
    Args:
        K_inv: inverse camera intrinsics matrix
        point: in image frame
        depth: depth associated with the given pixel
    Returns: 3x1 vector
    """
    return K_inv * [point...; 1] * depth
end

function get_point_3d(K_inv::SM3{Float32}, point::Tuple{Int, Int}, depth::Float32)::SV3{Float32}
    """
    Version of get_point_3d for integer points.
    Slow, use for visualization only.
    """
    return get_point_3d(K_inv, (Float32(point[1]), Float32(point[2])), depth)
end

function get_points_3d(K::SM3{Float32}, points, depth_map::Matrix{Float32})::Matrix{Float32}
    """
    Converts a list of points in the image frame to a 3d point cloud.
    Used for reprojecting keypoints to 3d space.
    Args:
        K: camera intrinsics matrix
        points: 2 x n array of points [u; v] to reproject
        depth_map: depth map for entire image
    Returns:
        3xN points
    """
    K_inv = SM3{Float32}(inv(K))
    n_pts = size(points, 2)
    UVD = Matrix{Float32}(undef, 3, n_pts)
    @inbounds for i in 1:n_pts
        point = points[:, i]
        # Round for indexing into the depth map. Note that keypoints are Float,
        # even though the image is composed of discrete pixels. Not sure if this
        # is the best way to reproject given the limitations of the depth map.
        # TODO(rgg): check if there's an off-by-one error from Python indexing
        u_ind, v_ind = round.(Int, point)
        d = depth_map[v_ind, u_ind]
        UVD[1, i] = point[1] * d
        UVD[2, i] = point[2] * d
        UVD[3, i] = d
        # Some will be projected to the origin (invalid depth=0)
        # We keep these in the output to match the input
    end
    return K_inv * UVD
end

function get_points_3d(K::SM3{Float32}, depth_map::Matrix{Float32})::Matrix{Float32}
    """
    Reproject all points in depth image to 3D.
    Used for reprojecting entire point cloud to 3d space for finding safe flight corridors.
    Args:
        K: camera intrinsics matrix
        depth_map: depth image
    Returns:
        Matrix{3, N, Float32} of xyz points where N is the number of pixels in the image
    """
    K_inv = inv(K) 
    N = prod(size(depth_map))
    UVD = Matrix{Float32}(undef, 3, N)
    tol = eps()
    ind = 1
    @inbounds for i in CartesianIndices(depth_map)
        d = depth_map[i]
        if abs(d) > tol  # invalid depth returns are mapped to the origin
            # u, v map to x, y; but cartesian indexing is [y][x]
            UVD[1, ind] = i[2] * d
            UVD[2, ind] = i[1] * d
            UVD[3, ind] = d
            ind += 1
        end
    end
    return K_inv * UVD[:, 1:ind-1]
end

# Functions for plotting / visualization
# TOOD(rgg): put these in separate file?

function get_point_color(point, color_img)
    """
    Args:
        point: [u, v], 1-indexed
        color_img: OpenCV image to extract color from
    Returns: RGB color from ColorTypes
    """
    # Point in [u, v]
    u_ind, v_ind = round.(Int, point)
    bgr = pyconvert(Vector{UInt32}, color_img[v_ind-1][u_ind-1]) ./ 255
    rgb = RGB(bgr[3], bgr[2], bgr[1])
    return rgb
end

line_material = LineBasicMaterial(color=RGB(0, 0, 1), linewidth=2.0)

function show_correspondence!(vis::Visualizer, match, kpoints1, kpoints2, img1_color, img2_color, points1_3d, points2_3d)
    """
    Shows all correspondences between two lists of 3d keypoints with colors from the image.
    Args:
        vis: a MeshCat Visualizer
        match: an opencv DMatch (through PythonCall)
        kpoints1: 2xN keypoints in [u, v] for image 1
        kpoints2: 2xN keypoints in [u, v] for image 2
        img1_color: RGB image 1
        img2_color: RGB image 2
        points1_3d: 3d keypoints for image 1
        points2_3d: 3d keypoints for image 2
    """
    # Extract the keypoints and adjust 0-indexing to 1-indexing
    idx1 = pyconvert(Int32, match.queryIdx)+1
    idx2 = pyconvert(Int32, match.trainIdx)+1
    pt1 = Point3f(points1_3d[:, idx1])
    pt2 = Point3f(points2_3d[:, idx2])
    # Check for correspondences to origin and skip
    if norm(pt1) < eps() || norm(pt2) < eps()
        return
    end

    line = [pt1, pt2]
    # Plot correspondence
    c1 = get_point_color(kpoints1[:, idx1], img1_color)
    c2 = get_point_color(kpoints2[:, idx2], img2_color)
    m1 = MeshLambertMaterial(color=c1)
    m2 = MeshLambertMaterial(color=c2)
    # Plot points
    setobject!(vis["pc1"][string(idx1)], Sphere(pt1, 0.001f0), m1)
    setobject!(vis["pc2"][string(idx2)], Sphere(pt2, 0.001f0), m2)
    # Connect with line
    line_str = "corrs" * string(idx1) * "_" * string(idx2)
    setobject!(vis["lines"][line_str], Object(PointCloud(line), line_material, "Line"))
end

function show_correspondence!(vis::Visualizer, kpoints1, kpoints2, label=string(uuid1()))
    """
    Shows all correspondences between two lists of 3d keypoints.
    Args:
        vis: a MeshCat Visualizer
        kpoints1: 3xN keypoints for image 1, columns aligned with kpoints2
        kpoints2: 3xN keypoints for image 2, columns aligned with kpoints1
    """
    # Extract the keypoints and adjust 0-indexing to 1-indexing
    N = size(kpoints1, 2)
    for i = 1:N
        pt1 = Point3f(kpoints1[:, i])
        pt2 = Point3f(kpoints2[:, i])
        # Do not check if correspondences are to origin, should be removed with remove_invalid_matches for this application
        line = [pt1, pt2]
        # TODO(recover color from original image?)
        c1 = RGB(0, 0, 1)
        if label == "gt"
            c2 = RGB(0, 1, 0)
        elseif label == "ls"
            c2 = RGB(1, 0.6, 0)
        elseif label == "invalid"
            c2 = RGB(1, 0, 0)
        else
            c2 = RGB(1, 1, 0)
        end
        # Plot correspondence
        m1 = MeshLambertMaterial(color=c1)
        m2 = MeshLambertMaterial(color=c2)
        # Plot points
        setobject!(vis["pc1_"*label][string(i)], Sphere(pt1, 0.001), m1)
        setobject!(vis["pc2_"*label][string(i)], Sphere(pt2, 0.001), m2)
        # Connect with line
        line_str = "corrs_" * label * "_" * string(i) * "_" * string(i)
        setobject!(vis["lines"][line_str], Object(PointCloud(line), line_material, "Line"))
    end
end

function show_pointcloud_color!(vis::Visualizer, depth_map, img_color, K, R=I, t=zeros(3), label::String=string(uuid1()))
    """
    Args:
        vis: MeshCat Visualizer
        depth_map: 2D indexable e.g. Matrix{Float64}
        img_color: 2D indexable Julia object (not Python)
        K: camera matrix
        R: [optional] rotation matrix to apply to every point
        t: [optional] translation to apply to every point
        label: [optional] unique identifier for pointcloud in MeshCat tree
    """
    colors = Colorant[]
    points = Point3f[]
    K_inv = inv(K)
    u_max = size(img_color, 2)
    v_max = size(img_color, 1)
    # TODO(rgg): clean this up with more efficient reprojection functions
    @inbounds for v in 1:v_max
        @inbounds for u in 1:u_max
            c = img_color[v, u]
            # This operation in particular can be done on an entire matrix of points for speedup
            pt3d = (R*get_point_3d(K_inv, (u, v), depth_map[v, u])) + t
            # Invalid returns will be exactly at the origin
            if norm(pt3d) > eps()
                push!(points, pt3d)
                push!(colors, c)
            end
        end
    end
    pc = PointCloud(points, colors)
    print("Plotting point cloud\n")
    # Need a unique name for each object or they will overwrite each other
    setobject!(vis["pc"][label], pc)  
    return label
end

function remove_invalid_matches(p1::Matrix, p2::Matrix)
    """
    Takes in two matching 3xN matrices of 3D point correspondences. 
    Removes all matching pairs where at least one point is at the origin.
    Points at the origin are indicative of invalid depth returns.
    Args:
        p1: 3xN matrix of 3D points
        p2: 3xN matrix of 3D points
    Returns: 
        Tuple(p1, p2): 3xM matrices of 3D points, where M <= N
    """
    n_pts = size(p1, 2)
    valid_inds = Int32[]
    # TODO(rgg): cleaner way with mapslices or similar?
    @inbounds for i in 1:n_pts
        pt1 = p1[:, i]
        pt2 = p2[:, i]
        if norm(pt1) > eps() && norm(pt2) > eps()
            push!(valid_inds, i)
        end
    end 
    return (p1[:, valid_inds], p2[:, valid_inds])
end

