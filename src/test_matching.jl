include("ingest.jl")
include("matching.jl")

# NOTE: test requires downloading "plant_4" eth3d dataset.
# To download, run:
# python3 download_eth3d_slam_datasets.py

function test_match(img1, img2, feature_string::String)
    kp1, kp2, matches = get_matches(img1, img2, feature_string)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[pyslice(10)],py.None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("/root/out/" * feature_string * "_matches.jpg", img3)
end

# Read in images
path1 = "/root/datasets/training/plant_4/rgb/4043.278005.png"
path2 = "/root/datasets/training/plant_4/rgb/4043.425458.png"
depth_path1 = "/root/datasets/training/plant_4/depth/4043.278005.png"
depth_path2 = "/root/datasets/training/plant_4/depth/4043.425458.png"

img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE) 
img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE) 
depth1 = cv.imread(depth_path1, cv.IMREAD_GRAYSCALE) 
depth2 = cv.imread(depth_path2, cv.IMREAD_GRAYSCALE) 

# Read in camera intrinsics matrix
cal_path = "/root/datasets/training/plant_4/calibration.txt"
K = assemble_K_matrix(get_cal_params(cal_path)...)

test_match(img1, img2, "orb")
test_match(img1, img2, "sift")
test_match(img1, img2, "akaze")
# test_match(img1, img2, "surf")  # Only if SURF is installed

# Test 3D correspondences
kp1, kp2, matches = get_matches(img1, img2, "orb")
# Get point cloud by reprojecting + depth info
# Invalid returns will be represented as NaN
# Assemple in numpy and convert to Julia. Not sure if optimal.
in_points1 = np.column_stack([kp.pt for kp in kp1]).T  
in_points1 = pyconvert(Matrix{Float64}, in_points1)
in_points2 = np.column_stack([kp.pt for kp in kp2]).T
in_points2 = pyconvert(Matrix{Float64}, in_points2)
depth1 = pyconvert(Matrix{UInt8}, depth1)  # Can we just read these in as Julia matrices?
depth2 = pyconvert(Matrix{UInt8}, depth2)
# Get 3D points for each keypoint in each frame
points1_3d = get_points_3d(K, in_points1, depth1)
points2_3d = get_points_3d(K, in_points2, depth2)
# Visualize