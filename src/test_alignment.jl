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
Integrated test to visualize alignment of point clouds from
two consecutive frames using transformation from PoseEstimation
"""
# Flag to test pose estimation with synthetic data.
# Not all plotting will work as expected with this.
use_synthetic_data = false

include("setup_test.jl")

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

# PoseEstimation expects two 3xN matrices
# First, get keypoint lists in 3d
p1 = np.column_stack([kp.pt for kp in kp1])  
p1 = pyconvert(Matrix{Float64}, p1)
p1_3d = get_points_3d(K, p1, depth1)

p2 = np.column_stack([kp.pt for kp in kp2])
p2 = pyconvert(Matrix{Float64}, p2)
p2_3d = get_points_3d(K, p2, depth2)

# Then, assemble corresponding matrices of points from matches
n_matches = pyconvert(Int32, py.len(matches))
matched_pts1 = zeros(Float64, 3, n_matches)
matched_pts2 = zeros(Float64, 3, n_matches)
for i in 1:n_matches
    m = matches[i-1]  # Python indexing
    idx1 = pyconvert(Int32, m.queryIdx)+1
    idx2 = pyconvert(Int32, m.trainIdx)+1
    matched_pts1[:, i] = p1_3d[idx1]
    matched_pts2[:, i] = p2_3d[idx2]
end
# Finally, clean correspondence list of any pairs that contain either point at the origin (invalid depth)
matched_pts1, matched_pts2 = remove_invalid_matches(matched_pts1, matched_pts2)

# Synthetic data
function generate_synthetic_data()
    Random.seed!(42);
    N = 50 # number of correspondences
    p1 = randn(3, N)
    # generate a ground-truth pose
    R_groundtruth = rand(RotMatrix{3})
    # generate a ground-truth translation
    t_groundtruth = randn(3)
    # generate true p2
    p2 = R_groundtruth * p1  .+ t_groundtruth
    # make noisy measurements
    β = 0.01
    p2_noisy = copy(p2)
    for i=1:N
        p2_noisy[:, i] += β*(2*rand(3).-1)
    end
    # add outliers to some% of data
    inds = [i for i=2:N if rand() < 0.10]
    for i=inds
        p2_noisy[:, i] += 3*randn(3)
    end
    return p1, p2_noisy, R_groundtruth, t_groundtruth
end
if use_synthetic_data
    matched_pts1, matched_pts2, R_gt_1_2, t_gt_1_2 = generate_synthetic_data()
end

# Compute R, t using TLS
c̄ = 0.07  # Maximum residual of inliers
@time R_tls_1_2, t_tls_1_2 = PE.estimate_Rt(matched_pts1, matched_pts2;
    method_pairing=PE.Star(),
    method_R=PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
    method_t=PE.TLS(c̄)
)
@time R_tls_2_1, t_tls_2_1 = PE.estimate_Rt(matched_pts2, matched_pts1;
    method_pairing=PE.Star(),
    method_R=PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
    method_t=PE.TLS(c̄)
)
T_tls_1_2 = get_T(R_tls_1_2, t_tls_1_2)
T_tls_2_1 = get_T(R_tls_2_1, t_tls_2_1)
@show norm(T_tls_1_2 - inv(T_tls_2_1))  # Should be 0

# Compute R, t using vanilla LS
R_ls_1_2, t_ls_1_2 = PE.estimate_Rt(matched_pts1, matched_pts2;
    method_pairing=PE.Star(),
    method_R=PE.LS(),
    method_t=PE.LS()
)
T_ls_1_2 = get_T(R_ls_1_2, t_ls_1_2)

# Get ground truth by interpolating
if !use_synthetic_data
    R_gt_1_2, t_gt_1_2 = get_groundtruth_Rt(gtruth, t1, t2)
    R_gt_2_1, t_gt_2_1 = get_groundtruth_Rt(gtruth, t2, t1)
end
T_gt_1_2 = get_T(R_gt_1_2, t_gt_1_2)
T_gt_2_1 = get_T(R_gt_2_1, t_gt_2_1)
@show norm(T_gt_1_2 - inv(T_gt_2_1))  # Should be 0

R_gt_1_w, t_gt_1_w = get_groundtruth_Rt(gtruth, t1)
R_gt_2_w, t_gt_2_w = get_groundtruth_Rt(gtruth, t2)
T_gt_1_w = get_T(R_gt_1_w, t_gt_1_w)
T_gt_2_w = get_T(R_gt_2_w, t_gt_2_w)

# Verify relative translation matches absolute
@show norm(T_gt_1_w - (T_gt_2_w*T_gt_1_2))  # Should be ~= 0

# Errors should be low
@show norm(T_gt_1_2 - T_tls_1_2)
@show norm(T_gt_2_1 - inv(T_tls_1_2))
@show norm(T_gt_2_1 - T_tls_2_1)
@show PE.rotdist(R_tls_1_2, R_gt_1_2) * 180 / π
@show norm(t_tls_1_2 - t_gt_1_2)

# Erros should be a bit higher
@show PE.rotdist(R_ls_1_2, R_gt_1_2) * 180 / π
@show norm(t_ls_1_2 - t_gt_1_2)

# Inlier noise, related to choice of ̄c. See TEASER paper.
β = 0.005  # TODO(rgg): refine this value and ̄c
# Estimate error on TLS
@show ϵR = PE.ϵR(matched_pts1, β)  # Requires only inliers, TODO(rgg): discuss
@show ϵt = PE.ϵt(β)


# Bring both sets of keypoints into the inertial frame
inertial_pts1 = apply_T(matched_pts1, T_gt_1_w)
inertial_pts2 = apply_T(matched_pts2, T_gt_2_w)

# Bring both sets of keypoints first into frame 2, then into inertial (world) frame
matched_pts1_rotated_tls = apply_T(matched_pts1, T_gt_2_w*T_tls_1_2)
matched_pts1_rotated_gt = apply_T(matched_pts1, T_gt_2_w*T_gt_1_2)
matched_pts1_rotated_ls = apply_T(matched_pts1, T_gt_2_w*T_ls_1_2)
# Plot both gt and tls points in world frame; should be close
show_correspondence!(vis, inertial_pts2, matched_pts1_rotated_gt, "gt")  # Green
show_correspondence!(vis, inertial_pts2, matched_pts1_rotated_tls, "tls")  # Yellow
# Show LS results (should be worse or equal to TLS)
show_correspondence!(vis, inertial_pts2, matched_pts1_rotated_ls, "ls")  # Yellow
# Show what would happen without correction:
# Interpret points in frame 1 erroneously as being in frame 2
show_correspondence!(vis, inertial_pts2, apply_T(matched_pts1, T_gt_2_w), "invalid")
