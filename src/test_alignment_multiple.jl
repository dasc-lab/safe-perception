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
Integrated test to visualize alignment of point clouds using computed 
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
gt_path = joinpath(df, "groundtruth.txt")
gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)

# Read in camera intrinsics matrix
cal_path = joinpath(df, "calibration.txt")
K = assemble_K_matrix(get_cal_params(cal_path)...)

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

c̄ = 0.07  # Maximum residual of inliers
# R1: inertial to frame 1.
# Inverse (transpose) of this should be frame 1 to inertial.

function plot_all()
    N = length(imgs_color)
    step = 3
    start = 9
    R_init, t_init = get_groundtruth_Rt(gtruth, depth_ts[start])
    local prev_T = get_T(R_init, t_init)
    show_pointcloud_color!(vis, dimgs[start], imgs_color[start], K, R_init, t_init)
    for i in start:step:N-step
        t_1 = depth_ts[i]
        t_2 = depth_ts[i+step]
        matched_pts1, matched_pts2 = get_matched_pts(imgs_gray[i], imgs_gray[i+step], dimgs[i], dimgs[i+step])
        R_tls_2_1, t_tls_2_1 = PE.estimate_Rt(matched_pts2, matched_pts1;
            method_pairing=PE.Star(),
            method_R=PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
            method_t=PE.TLS(c̄)
        )
        T_tls_2_1 = get_T(R_tls_2_1, t_tls_2_1)
        R_gt_2_1, t_gt_2_1 = get_groundtruth_Rt(gtruth, t_2, t_1)
        T_gt_2_1 = get_T(R_gt_2_1, t_gt_2_1)
        @show norm(T_gt_2_1 - T_tls_2_1)
        T_gt_2_w = get_T(get_groundtruth_Rt(gtruth, t_2)...)
        prev_T = prev_T * T_tls_2_1
        @show norm(prev_T - T_gt_2_w)
        R, t = extract_R_t(prev_T)
        show_pointcloud_color!(vis, dimgs[i+step], imgs_color[i+step], K, R, t)
    end
end

plot_all()