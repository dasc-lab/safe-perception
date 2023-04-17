import Pkg
using Logging, Printf
logger = Logging.ConsoleLogger(Logging.Info)
Logging.global_logger(logger)

include("PoseEstimation.jl")
include("ingest.jl")
include("matching.jl")
include("mapping.jl")
include("utils.jl")
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

# Read in groundtruth
# Columns: timestamp tx ty tz qx qy qz qw
const gt_path = joinpath(df, "groundtruth.txt")
const gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)

# Read in camera intrinsics matrix
const cal_path = joinpath(df, "calibration.txt")
const K = SM3{Float32}(assemble_K_matrix(get_cal_params(cal_path)...))

# Whether to visualize the point clouds and polyhedra in MeshCat
visualize = false

if visualize
    if !@isdefined vis
        vis = Visualizer()  # Only need to set up once
    end
    delete!(vis)  # Clear any rendered objects
end

const c̄ = 1f0  # Maximum residual of inliers
const β = 0.005f0  # TODO(rgg): refine this value and ̄c

function run_test()
    """
    Iterates over some number of frames and computes rototranslation using
    visual odometry. Computes safe flight corridor polyhedra.
    """
    N = length(img_filenames)
    step = 3
    start = 9
    stop = 180
    R_init, t_init = get_groundtruth_Rt(gtruth, depth_ts[start])
    local prev_T = get_T(R_init, t_init)

    # Get next two frames
    curr_dimg = get_depth(df, depth_filenames[start])
    curr_color = get_imgs(df, img_filenames[start])
    curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)

    # Extract image dimensions (assume they do not change)
    yrange = [0, pyconvert(Int32, curr_color.shape[0])]
    xrange = [0, pyconvert(Int32, curr_color.shape[1])]

    if visualize
        # Show first frame
        curr_color_jl = convert_py_rgb_img(curr_color)
        show_pointcloud_color!(vis, curr_dimg, curr_color_jl, K, R_init, t_init)
    end

    for i in start:step:stop
        @info @sprintf("\n============== Step %i of %i ===============\n", i, (N-step))
        @time begin 
            curr_dimg = get_depth(df, depth_filenames[i])
            next_dimg = get_depth(df, depth_filenames[i+step])
            curr_color = get_imgs(df, img_filenames[i])
            next_color = get_imgs(df, img_filenames[i+step])
            curr_gray = cv.cvtColor(curr_color, cv.COLOR_BGR2GRAY)
            next_gray = cv.cvtColor(next_color, cv.COLOR_BGR2GRAY)

            t_1 = depth_ts[i]
            t_2 = depth_ts[i+step]
            @info "Finding correspondences"
            @time matched_pts1, matched_pts2 = get_matched_pts(curr_gray, next_gray, curr_dimg, next_dimg)
            @info "Estimating R, t with TLS"
            @time R_tls_2_1, t_tls_2_1 = PE.estimate_Rt_fast(matched_pts2, matched_pts1;
                β=β,
                method_pairing=PE.Star(),
                method_R=PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
                method_t=PE.TLS(c̄)
            )
            T_tls_2_1 = get_T(R_tls_2_1, t_tls_2_1)  # Transforms points in frame 2 to frame 1
            #R_gt_2_1, t_gt_2_1 = get_groundtruth_Rt(gtruth, t_2, t_1)
            #T_gt_2_1 = get_T(R_gt_2_1, t_gt_2_1)
            #@show norm(T_gt_2_1 - T_tls_2_1)
            #T_gt_2_w = get_T(get_groundtruth_Rt(gtruth, t_2)...)
            prev_T = prev_T * T_tls_2_1
            #@show norm(prev_T - T_gt_2_w)

            # Calculate error bounds for relative transformation (not cumulative)
            # Inlier noise, related to choice of ̄c. See TEASER paper.
            # Estimate error on TLS
            @info "Estimating error bounds with max clique and sampling"
            @time ϵR, ϵt = PE.est_err_bounds(matched_pts1, matched_pts2, β, iterations=10000)
            @show ϵR, ϵt
            max_dist = 1f0 # maximum distance from camera to any point [m]
            @show norm_ball_err = ϵR * max_dist + ϵt

            R, t = extract_R_t(prev_T)
            seed = SV3([0f0, 0f0, 0f0])
            @info "Reprojecting depth image and transforming"
            @time obs_points_camera_frame = get_points_3d(K, next_dimg)
            # TODO(rgg): add norm ball errors
            translucent_purple = MeshLambertMaterial(color=RGBA(0.5, 0, 0.5, 0.5))
            translucent_red = MeshLambertMaterial(color=RGBA(1, 0, 0, 0.5))
            @info "Getting obstacle-free polyhedron with DecompUtil"
            # Will only be guaranteed to not intersect obstacles in the current frame
            @time obs_poly = get_obs_free_polyhedron(obs_points_camera_frame,
                                                     seed,
                                                     T=inv(prev_T),
                                                     bbox=[2, 2, 2],
                                                     ϵ=norm_ball_err)
            @info "Getting FOV poly\n"
            @time fov_poly = get_fov_polyhedron(K, inv(prev_T), xrange, yrange)
            safe_poly = intersect(fov_poly, obs_poly)

            if visualize
                safe_poly_mesh = Polyhedra.Mesh(safe_poly)
                obs_poly_mesh = Polyhedra.Mesh(obs_poly)
                # Convert Python color image to julia matrix
                next_color_jl = convert_py_rgb_img(next_color)
                show_pointcloud_color!(vis, next_dimg, next_color_jl, K, R, t)
                # show_pointcloud_color!(vis, next_dimg, next_color_jl, K, I(3), [0.;0;0])
                setobject!(vis["safe_poly"], safe_poly_mesh, translucent_red)
                # plot_fov_polyhedron!(vis, K, inv(prev_T), xrange, yrange, max_depth=3)
                # setobject!(vis["obs_poly"], obs_poly_mesh, translucent_purple)
            end
            @info "Completed processing one frame"
        end
    end
end