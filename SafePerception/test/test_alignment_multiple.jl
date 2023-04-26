include("PoseEstimation.jl")
include("ingest.jl")
include("matching.jl")
include("utils.jl")
using .PoseEstimation
using BenchmarkTools, Random, Rotations, Interpolations, DelimitedFiles
using PythonCall
using Printf
using DecompUtil
PE = PoseEstimation
py = pybuiltins

"""
Integrated test to visualize alignment of point clouds using computed 
rototranslation.
"""
# Tested with plant_4, mannequin_face_1, sofa_2, plant_scene_2
df = joinpath("/root/datasets/training/sofa_2/")

# Read in images
ground_truth = readdlm(joinpath(df, "groundtruth.txt"), skipstart=1);
depth_filenames = readdlm(joinpath(df, "depth.txt"));
depth_ts = depth_filenames[:,1]
depth_filenames = depth_filenames[:,2]  # Remove other columns for convenience
img_filenames = [joinpath("rgb", f) for f in readdir(joinpath(df, "rgb"))]

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

const c̄ = 1f0  # Maximum residual of inliers
const β = 0.005f0  # Bound on inlier noise

function plot_all()
    N = length(img_filenames)
    step = 3
    start = 9
    stop = 150
    R_init, t_init = get_groundtruth_Rt(gtruth, depth_ts[start])
    local prev_T = get_T(R_init, t_init)

    # Get next two frames
    curr_dimg = get_depth(df, depth_filenames[start])
    curr_color = get_imgs(df, img_filenames[start])
    curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)

    curr_color_jl = convert_py_rgb_img(curr_color)
    show_pointcloud_color!(vis, curr_dimg, curr_color_jl, K, R_init, t_init)
    for i in start:step:stop
        @printf "Step %i of %i\n" i (N-step)
        curr_dimg = get_depth(df, depth_filenames[i])
        next_dimg = get_depth(df, depth_filenames[i+step])
        curr_color = get_imgs(df, img_filenames[i])
        next_color = get_imgs(df, img_filenames[i+step])
        curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)
        next_gray = cv.cvtColor(next_color, cv.COLOR_BGR2GRAY)

        t_1 = depth_ts[i]
        t_2 = depth_ts[i+step]
        global β
        matched_pts1, matched_pts2 = get_matched_pts(curr_gray, next_gray, curr_dimg, next_dimg)
        @time R_tls_2_1, t_tls_2_1 = PE.estimate_Rt(matched_pts2, matched_pts1;
            method_pairing=PE.Star(),
            β = β,
            method_R=PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
            method_t=PE.TLS(c̄)
        )
        T_tls_2_1 = get_T(R_tls_2_1, t_tls_2_1)
        R_gt_2_1, t_gt_2_1 = get_groundtruth_Rt(gtruth, t_2, t_1)
        T_gt_2_1 = get_T(R_gt_2_1, t_gt_2_1)
        #@show norm(T_gt_2_1 - T_tls_2_1)
        T_gt_2_w = get_T(get_groundtruth_Rt(gtruth, t_2)...)
        prev_T = prev_T * T_tls_2_1
        #@show norm(prev_T - T_gt_2_w)

        # Calculate error bounds for relative transformation (not cumulative)
        # Estimate error on TLS
        @time ϵR, ϵt = PE.est_err_bounds(matched_pts1, matched_pts2, β, iterations=1000)
        @show ϵR, ϵt

        R, t = extract_R_t(prev_T)
        next_color_jl = convert_py_rgb_img(next_color)
        show_pointcloud_color!(vis, next_dimg, next_color_jl, K, R, t)
    end
end

plot_all()