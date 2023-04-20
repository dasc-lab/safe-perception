"""
Common test setup with two frames.
Loads data and computes keypoints.
"""
# Select dataset and frames as desired
df_path = "/root/datasets/training/plant_4/"
ind1 = 47  # Index of first frame
ind2 = 50  # Index of second frame

function name_to_timestamp(name)
    parts = split(name, '.')
    t_str = parts[1] * "." * parts[2]
    return parse(Float64, t_str)
end
filenames = readdir(joinpath(df_path, "rgb"))
name1 = filenames[ind1]
name2 = filenames[ind2]
t1 = name_to_timestamp(name1)
t2 = name_to_timestamp(name2)

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