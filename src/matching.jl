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

    Arguments
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
        # Note: SURF is patented and requires workarounds when building/installing OpenCV
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
