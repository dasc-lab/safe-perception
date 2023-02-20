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
data_folder = joinpath("/root/datasets/training/plant_4/")

function get_depth(dimg_name)
    depth_path = joinpath(data_folder, dimg_name) 
    depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH) 
    depth = pyconvert(Matrix{UInt16}, depth) ./ 5000  # Divide by 5000 for eth3d dataset
    return depth
end

function get_imgs(img_name)
    path = joinpath(data_folder, img_name) 
    img_color = cv.imread(path, cv.IMREAD_COLOR) 
    return img_color
end

# Read in images
ground_truth = readdlm(joinpath(data_folder, "groundtruth.txt"), skipstart=1);
depth_filenames = readdlm(joinpath(data_folder, "depth.txt"));
depth_ts = depth_filenames[:,1]
dimgs = [get_depth(n) for n in depth_filenames[:,2]]
img_filenames = [joinpath("rgb", f) for f in readdir(joinpath(data_folder, "rgb"))]
imgs_color = [get_imgs(n) for n in img_filenames]
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
for i in 1:3:N
    R, t = get_groundtruth_Rt(gtruth, depth_ts[i])
    show_pointcloud_color!(vis, dimgs[i], imgs_color[i], K, R, t)
end