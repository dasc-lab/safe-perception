# Make sure we use the system-installed Python backend
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/usr/bin/python3"

using PythonCall

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

function get_point_3d(K_inv::Matrix{Float64}, point::Vector, depth)
    """
    Takes a point in u,v and returns a point in x, y, z using the inverse
    camera intrinsics matrix and the known depth (z) of the point.

    Args:
        K_inv: inverse camera intrinsics matrix
        point: in image frame
        depth: depth associated with the given pixel
    """
    point_3d = K_inv * [point; 1] * depth
    return point_3d
end

function get_points_3d(K, points, depth_map)
    """
    Converts a list of points in the image frame to a 3d point cloud.

    Args:
        K: camera intrinsics matrix
        points: n x 2 array of points [u, v] to reproject
        depth_map: depth map for entire image
    """
    K_inv = inv(K)
    n_pts = size(points, 1)
    out_points = zeros(n_pts, 3)
    for i in 1:n_pts
        point = points[i, :]
        # Round for indexing into the depth map. Note that keypoints are Float64,
        # even though the image is composed of discrete pixels. Not sure if this
        # is the best way to reproject given the limitations of the depth map.
        u_ind, v_ind = round.(Int, point)
        out_points[i, :] = get_point_3d(K_inv, point, depth_map[v_ind, u_ind])  # TODO(rgg): check this indexing is correct
    end
    return out_points
end

