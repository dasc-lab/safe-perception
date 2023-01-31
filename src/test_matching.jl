include("ingest.jl")
include("matching.jl")

# Read in images
path1 = "../datasets/training/plant_4/rgb/4043.278005.png"
path2 = "../datasets/training/plant_4/rgb/4043.425458.png"
depth_path1 = "../datasets/training/plant_4/depth/4043.278005.png"
depth_path2 = "../datasets/training/plant_4/depth/4043.425458.png"

img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE) 
img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE) 
depth1 = cv.imread(depth_path1, cv.IMREAD_GRAYSCALE) 
depth2 = cv.imread(depth_path2, cv.IMREAD_GRAYSCALE) 

# Read in camera intrinsics matrix
cal_path = "../datasets/training/plant_4/calibration.txt"
K = assemble_K_matrix(get_cal_params(cal_path)...)

kp1, kp2, matches = get_matches(img1, img2, "orb")
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[pyslice(10)],py.None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite("orb_matches.jpg", img3)

kp1, kp2, matches = get_matches(img1, img2, "sift")
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[pyslice(10)],py.None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite("sift_matches.jpg", img3)

kp1, kp2, matches = get_matches(img1, img2, "akaze")
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[pyslice(10)],py.None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imwrite("akaze_matches.jpg", img3)

# Get point cloud by reprojecting + depth info
# Invalid returns will be represented as NaN
points1 = cv.rgbd.depthTo3d(np.uint16(depth2), np.array(K))
points2 = cv.rgbd.depthTo3d(np.uint16(depth2), np.array(K))
# Get 3D points for each keypoint in each frame
