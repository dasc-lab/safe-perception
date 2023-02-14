include("PoseEstimation.jl")
include("ingest.jl")
include("matching.jl")
include("test_utils.jl")
using .PoseEstimation
using BenchmarkTools, Random, Rotations, Interpolations, DelimitedFiles
using PythonCall
PE = PoseEstimation
py = pybuiltins

"""
Integrated test to visualize alignment of point clouds using ground truth
rototranslation.
"""
function get_depth(t_str)
    depth_path = "/root/datasets/training/plant_4/depth/" * t_str * ".png"
    depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH) 
    depth = pyconvert(Matrix{UInt16}, depth) ./ 5000  # Divide by 5000 for eth3d dataset
    return depth
end

function get_imgs(t_str)
    path = "/root/datasets/training/plant_4/rgb/" * t_str * ".png"
    img_color = cv.imread(path, cv.IMREAD_COLOR) 
    img = cv.cvtColor(img_color, cv.COLOR_BGR2GRAY)
    return img, img_color
end

# Read in images
t1_str = "4043.278005"
t2_str = "4043.314868"
t3_str = "4044.789402"
t1 = parse(Float64, t1_str)
t2 = parse(Float64, t2_str)
t3 = parse(Float64, t3_str)

img1, img1_color = get_imgs(t1_str) 
img2, img2_color = get_imgs(t2_str) 
img3, img3_color = get_imgs(t3_str) 

depth1 = get_depth(t1_str)
depth2 = get_depth(t2_str)
depth3 = get_depth(t3_str)

# Read in groundtruth
# Columns: timestamp tx ty tz qx qy qz qw
gt_path = "/root/datasets/training/plant_4/groundtruth.txt"
gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)
# Get ground truth by interpolating
R_gt, t_gt = get_groundtruth_Rt(gtruth, t1, t2)

# Read in camera intrinsics matrix
cal_path = "/root/datasets/training/plant_4/calibration.txt"
K = assemble_K_matrix(get_cal_params(cal_path)...)

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

# R1: inertial to frame 1.
# Inverse (transpose) of this should be frame 1 to inertial.
R1, trans1 = get_groundtruth_Rt(gtruth, t1)
R2, trans2 = get_groundtruth_Rt(gtruth, t2)
R3, trans3 = get_groundtruth_Rt(gtruth, t3)
show_pointcloud_color!(vis, depth1, img1_color, K, inv(R1), -trans1)
show_pointcloud_color!(vis, depth2, img2_color, K, inv(R2), -trans2)
show_pointcloud_color!(vis, depth3, img3_color, K, inv(R3), -trans3)