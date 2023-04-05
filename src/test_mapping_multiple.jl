import Pkg
# run(`ls`)
# Pkg.activate("src/")
# Pkg.instantiate()
include("PoseEstimation.jl")
include("ingest.jl")
include("matching.jl")
include("mapping.jl")
include("test_utils.jl")
using .PoseEstimation
using BenchmarkTools, Random, Rotations, Interpolations, DelimitedFiles
using PythonCall
using Printf
using DecompUtil
PE = PoseEstimation
py = pybuiltins

"""
Integrated test to visualize safe flight corridors computed on 
point clouds with TLS rototranslation.
"""
# Tested with plant_4, mannequin_face_1, sofa_2, plant_scene_2
const df = joinpath("/root/datasets/training/sofa_2/")

# Read in images
const ground_truth = readdlm(joinpath(df, "groundtruth.txt"), skipstart=1);
const depth_info = readdlm(joinpath(df, "depth.txt"));
const depth_ts = depth_info[:,1]
const depth_filenames = depth_info[:,2]  # Remove other columns for convenience
const img_filenames = [joinpath("rgb", f) for f in readdir(joinpath(df, "rgb"))]

#dimgs = [get_depth(df, n) for n in depth_filenames[:,2]]
#imgs_color = [get_imgs(df, n) for n in img_filenames]
#imgs_gray = [cv.cvtColor(ic, cv.COLOR_BGR2GRAY) for ic in imgs_color]  # Used for keypoints

# Read in groundtruth
# Columns: timestamp tx ty tz qx qy qz qw
const gt_path = joinpath(df, "groundtruth.txt")
const gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)

# Read in camera intrinsics matrix
const cal_path = joinpath(df, "calibration.txt")
const K = SM3{Float32}(assemble_K_matrix(get_cal_params(cal_path)...))

# if !@isdefined vis
#     vis = Visualizer()  # Only need to set up once
# end
# delete!(vis)  # Clear any rendered objects

const c̄ = 0.07  # Maximum residual of inliers

function plot_all()
    N = length(img_filenames)
    step = 3
    start = 9
    stop = 27
    R_init, t_init = get_groundtruth_Rt(gtruth, depth_ts[start])
    local prev_T = get_T(R_init, t_init)

    # Get next two frames
    curr_dimg = get_depth(df, depth_filenames[start])
    curr_color = get_imgs(df, img_filenames[start])
    curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)

    # Extract image dimensions (assume they do not change)
    yrange = [0, pyconvert(Int32, curr_color.shape[0])]
    xrange = [0, pyconvert(Int32, curr_color.shape[1])]

    #show_pointcloud_color!(vis, curr_dimg, curr_color, K, R_init, t_init)
    for i in start:step:stop
        @printf "\n============== Step %i of %i ===============\n" i (N-step)
        @time begin 
            curr_dimg = get_depth(df, depth_filenames[i])
            next_dimg = get_depth(df, depth_filenames[i+step])
            curr_color = get_imgs(df, img_filenames[i])
            next_color = get_imgs(df, img_filenames[i+step])
            curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(next_color, cv.COLOR_BGR2GRAY)

            t_1 = depth_ts[i]
            t_2 = depth_ts[i+step]
            @printf "Finding correspondences\n"
            @time matched_pts1, matched_pts2 = get_matched_pts(curr_gray, next_gray, curr_dimg, next_dimg)
            @printf "Estimating R, t with TLS"
            @time R_tls_2_1, t_tls_2_1 = PE.estimate_Rt(matched_pts2, matched_pts1;
                method_pairing=PE.Star(),
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
            # Inlier noise, related to choice of ̄c. See TEASER paper.
            β = 0.005  # TODO(rgg): refine this value and ̄c
            # Estimate error on TLS
            @printf "Estimating error bounds with max clique and sampling\n"
            @time ϵR, ϵt = PE.est_err_bounds(matched_pts1, matched_pts2, β, iterations=1000)
            @show ϵR, ϵt

            R, t = extract_R_t(prev_T)
            seed = apply_T([0; 0; 0.0], prev_T)
            @printf "Reprojecting depth image and transforming"
            @time obs_points_camera_frame = get_points_3d(K, next_dimg)
            # TODO(rgg): add norm ball errors
            translucent_purple = MeshLambertMaterial(color=RGBA(0.5, 0, 0.5, 0.5))
            translucent_red = MeshLambertMaterial(color=RGBA(1, 0, 0, 0.5))
            @printf "Getting obstacle-free polyhedron with DecompUtil\n"
            @time obs_poly = get_obs_free_polyhedron(obs_points_camera_frame, seed, T=inv(prev_T), bbox=[3, 3, 3])
            fov_poly = get_fov_polyhedron(K, inv(prev_T), xrange, yrange)
            #safe_poly = intersect(fov_poly, obs_poly)
            #safe_poly_mesh = Polyhedra.Mesh(safe_poly)
            #obs_poly_mesh = Polyhedra.Mesh(obs_poly)

            #show_pointcloud_color!(vis, next_dimg, next_color, K, R, t)
            #setobject!(vis["safe_poly"], safe_poly_mesh, translucent_red)
            #plot_fov_polyhedron!(vis, K, inv(prev_T), xrange, yrange, max_depth=3)
            #setobject!(vis["obs_poly"], obs_poly_mesh, translucent_purple)
        end
    end
end

plot_all()