# Make sure we use the system-installed Python backend
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/usr/bin/python3"

using PythonCall, GeometryBasics, LinearAlgebra
using ColorTypes, MeshCat

cv = pyimport("cv2")
np = pyimport("numpy")
py = pybuiltins

function get_matches(img1, img2, detector_type::String="orb")
    """
    Get matches between two images.

    Args:
        img1: first image
        img2: second image 
        detector_type: "orb", "sift", or "akaze"

    Returns: keypoints from image 1, keypoints from image 2, matches, all as relevant OpenCV objects
    """

    # Set up detector and distance to use for matching
    if detector_type == "orb"
        detector = cv.ORB_create()
        feature_norm = cv.NORM_HAMMING
    elseif detector_type == "sift"
        # May not work depending on version of OpenCV; patent expired 03-2020
        detector = cv.SIFT_create()
        feature_norm = cv.NORM_L2
    elseif detector_type == "akaze"
        detector = cv.AKAZE_create()
        feature_norm = cv.NORM_HAMMING
    elseif detector_type == "surf"
        # Note: SURF is patented and requires a special build flag / contrib opencv
        detector = cv.SURF_create()
        feature_norm = cv.NORM_L2
    else
        return
    end

    # Find the keypoints and descriptors.
    kp1, des1 = detector.detectAndCompute(img1, py.None)
    kp2, des2 = detector.detectAndCompute(img2, py.None)
    
    # TODO(rgg): support FLANN, see https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    # Create BFMatcher object. Hamming distance for ORB, L2 for SIFT/SURF.
    bf = cv.BFMatcher(feature_norm, crossCheck=true)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # Sort them in the order of their distance.
    matches = py.sorted(matches, key = x -> x.distance)

    return kp1, kp2, matches
end

function get_point_3d(K_inv::Matrix{Float64}, point, depth)
    """
    Takes a point in u,v and returns a point in x, y, z using the inverse
    camera intrinsics matrix and the known depth (z) of the point.

    Args:
        K_inv: inverse camera intrinsics matrix
        point: in image frame
        depth: depth associated with the given pixel
    Returns: Point3f
    """
    point_3d = K_inv * [point...; 1] * depth
    return Point3f(point_3d)
end

function get_points_3d(K, points, depth_map)
    """
    Converts a list of points in the image frame to a 3d point cloud.

    Args:
        K: camera intrinsics matrix
        points: 2 x n array of points [u; v] to reproject
        depth_map: depth map for entire image
    Returns:
        Vector{Point3f}
    """
    K_inv = inv(K)
    n_pts = size(points, 2)
    out_points = Point3f[]
    for i in 1:n_pts
        point = points[:, i]
        # Round for indexing into the depth map. Note that keypoints are Float64,
        # even though the image is composed of discrete pixels. Not sure if this
        # is the best way to reproject given the limitations of the depth map.
        u_ind, v_ind = round.(Int, point)
        pt_3d = get_point_3d(K_inv, point, depth_map[v_ind, u_ind])
        # Some will be projected to the origin (invalid depth)
        # We keep these in the output to match the input
        push!(out_points, pt_3d)
    end
    return out_points
end

function get_points_3d(K, depth_map)
    K_inv = inv(K)
    out_points = Point3f[]
    c_inds = CartesianIndices(depth_map)
    for i in c_inds
        pt_3d = get_point_3d(K_inv, Tuple(i), depth_map[i])
        # Invalid returns will be exactly at the origin, remove
        if norm(pt_3d) > eps()
            push!(out_points, pt_3d)
        end
    end
    return out_points
end

# Functions for plotting / visualization
# TOOD(rgg): put in separate file?

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
    pt1 = points1_3d[idx1]
    pt2 = points2_3d[idx2]
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
    setobject!(vis["pc1"][string(idx1)], Sphere(pt1, 0.001), m1)
    setobject!(vis["pc2"][string(idx2)], Sphere(pt2, 0.001), m2)
    # Connect with line
    line_str = "corrs" * string(idx1) * "_" * string(idx2)
    setobject!(vis["lines"][line_str], Object(PointCloud(line), line_material, "Line"))
end

function show_correspondence!(vis::Visualizer, kpoints1, kpoints2, label="default")
    """
    Shows all correspondences in matches lists of 3d keypoints.
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

function show_pointcloud_color!(vis, depth_map, img_color, K)
    colors = Colorant[]
    points = Point3f[]
    K_inv = inv(K)
    u_max = pyconvert(Int32, np.shape(img_color)[1])
    v_max = pyconvert(Int32, np.shape(img_color)[0])
    for v in 1:v_max
        for u in 1:u_max
            c = get_point_color([u; v], img_color)
            pt3d = get_point_3d(K_inv, [u; v], depth_map[v, u])
            # Invalid returns will be exactly at the origin
            if norm(pt3d) > eps()
                push!(points, pt3d)
                #push!(points, Point3f([u; v; 0] ./ 600))
                push!(colors, c)
            end
        end
    end
    pc = PointCloud(points, colors)
    print("Plotting point cloud\n")
    setobject!(vis["pc"], pc)
end

function remove_invalid_matches(p1, p2)
    """
    Takes in two matching 3xN matrices of 3D point correspondences. 
    Removes all matching pairs where at least one point is at the origin.
    Points at the origin are indicative of invalid depth returns.
    """
    n_pts = size(p1, 2)
    valid_inds = Int32[]
    # TODO(rgg): cleaner way with mapslices or similar?
    for i in 1:n_pts
        pt1 = p1[:, i]
        pt2 = p2[:, i]
        if norm(pt1) > eps() && norm(pt2) > eps()
            push!(valid_inds, i)
        end
    end 
    return (p1[:, valid_inds], p2[:, valid_inds])
end