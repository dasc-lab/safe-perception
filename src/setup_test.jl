"""
Common test setup with two frames.
Loads data and computes keypoints.
"""
t1_str = "4043.278005"
t2_str = "4043.314868"
t1 = parse(Float64, t1_str)
t2 = parse(Float64, t2_str)
path1 = "/root/datasets/training/plant_4/rgb/" * t1_str * ".png"
path2 = "/root/datasets/training/plant_4/rgb/" * t2_str * ".png"
depth_path1 = "/root/datasets/training/plant_4/depth/" * t1_str * ".png"
depth_path2 = "/root/datasets/training/plant_4/depth/" * t2_str * ".png"

img1_color = cv.imread(path1, cv.IMREAD_COLOR) 
img2_color = cv.imread(path2, cv.IMREAD_COLOR)
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

depth1 = cv.imread(depth_path1, cv.IMREAD_ANYDEPTH) 
depth2 = cv.imread(depth_path2, cv.IMREAD_ANYDEPTH) 
depth1 = pyconvert(Matrix{UInt16}, depth1) ./ 5000  # Divide by 5000 for eth3d dataset
depth2 = pyconvert(Matrix{UInt16}, depth2) ./ 5000

# Read in groundtruth
# Columns: timestamp tx ty tz qx qy qz qw
gt_path = "/root/datasets/training/plant_4/groundtruth.txt"
gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)

# Read in camera intrinsics matrix
cal_path = "/root/datasets/training/plant_4/calibration.txt"
K = assemble_K_matrix(get_cal_params(cal_path)...)

kp1, kp2, matches = get_matches(img1, img2, "orb")