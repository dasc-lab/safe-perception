include("PoseEstimation.jl")
include("ingest.jl")
include("matching.jl")
include("utils.jl")
using .PoseEstimation
using BenchmarkTools, Random, Rotations, Interpolations, DelimitedFiles
using PythonCall
PE = PoseEstimation
py = pybuiltins

"""
Integrated test to visualize alignment of point clouds using ground truth
rototranslation.
"""
df = joinpath("/root/datasets/training/plant_4/")

# Read in images
ground_truth = readdlm(joinpath(df, "groundtruth.txt"), skipstart=1);
depth_filenames = readdlm(joinpath(df, "depth.txt"));
depth_ts = depth_filenames[:,1]
dimgs = [get_depth(df, n) for n in depth_filenames[:,2]]
img_filenames = [joinpath("rgb", f) for f in readdir(joinpath(df, "rgb"))]
imgs_color = [get_imgs(df, n) for n in img_filenames]
imgs_gray = [cv.cvtColor(ic, cv.COLOR_BGR2GRAY) for ic in imgs_color]  # Used for keypoints

# Read in groundtruth
# Columns: timestamp tx ty tz qx qy qz qw
gt_path = "/root/datasets/training/plant_4/groundtruth.txt"
gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)

# Read in camera intrinsics matrix
cal_path = "/root/datasets/training/plant_4/calibration.txt"
K = assemble_K_matrix(get_cal_params(cal_path)...)

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

# R1: inertial to frame 1.
# Inverse (transpose) of this should be frame 1 to inertial.
N = length(imgs_color)
for i in 4:3:N-3
    R, t = get_groundtruth_Rt(gtruth, depth_ts[i])
    show_pointcloud_color!(vis, dimgs[i], imgs_color[i], K, R, t)
end