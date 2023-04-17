"""
Common test setup with two frames.
Loads data and computes keypoints.
"""
# Select dataset and frames as desired
df_path = "/root/datasets/training/plant_4/"
t1_str = "4043.278005"
t2_str = "4043.314868"

t1 = parse(Float64, t1_str)
t2 = parse(Float64, t2_str)
name1 = t1_str * ".png"
name2 = t2_str * ".png"

rgb_path = joinpath(df_path, "rgb/")
img1_color = get_imgs(rgb_path, name1) 
img2_color = get_imgs(rgb_path, name2) 
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

depth_path = joinpath(df_path, "depth/")
depth1 = get_depth(depth_path, name1)
depth2 = get_depth(depth_path, name2)

# Read in groundtruth
# Columns: timestamp tx ty tz qx qy qz qw
gt_path = joinpath(df_path, "groundtruth.txt")
gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)

# Read in camera intrinsics matrix
cal_path = joinpath(df_path, "calibration.txt")
K = assemble_K_matrix(get_cal_params(cal_path)...)

kp1, kp2, matches = get_matches(img1, img2, "orb")