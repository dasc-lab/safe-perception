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

# Read in images
t1_str = "4043.278005"
t2_str = "4043.314868"
t1 = parse(Float64, t1_str)
t2 = parse(Float64, t2_str)
path1 = "/root/datasets/training/plant_4/rgb/" * t1_str * ".png"
path2 = "/root/datasets/training/plant_4/rgb/" * t2_str * ".png"
depth_path1 = "/root/datasets/training/plant_4/depth/" * t1_str * ".png"
depth_path2 = "/root/datasets/training/plant_4/depth/" * t2_str * ".png"

img1_color = cv.imread(path1, cv.IMREAD_COLOR) 
img2_color = cv.imread(path2, cv.IMREAD_COLOR)
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

depth1 = cv.imread(depth_path1, cv.IMREAD_ANYDEPTH) 
depth2 = cv.imread(depth_path2, cv.IMREAD_ANYDEPTH) 
depth1 = pyconvert(Matrix{UInt16}, depth1) ./ 5000  # Divide by 5000 for eth3d dataset
depth2 = pyconvert(Matrix{UInt16}, depth2) ./ 5000

# Read in groundtruth
# Columns: timestamp tx ty tz qx qy qz qw
gt_path = "/root/datasets/training/plant_4/groundtruth.txt"
gtruth = readdlm(gt_path, ' ', Float64, skipstart=1)
# Get ground truth by interpolating
R_gt, t_gt = get_groundtruth_Rt(gtruth, t1, t2)

# Read in camera intrinsics matrix
cal_path = "/root/datasets/training/plant_4/calibration.txt"
K = assemble_K_matrix(get_cal_params(cal_path)...)

kp1, kp2, matches = get_matches(img1, img2, "orb")

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
# Uncomment to use synthetic data
#matched_pts1, matched_pts2, R_gt, t_gt = generate_synthetic_data()

# Compute R, t using TLS
c̄ = 0.07  # Maximum residual of inliers
@time R_tls, t_tls = PE.estimate_Rt(matched_pts1, matched_pts2;
    method_pairing=PE.Star(),
    method_R=PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
    method_t=PE.TLS(c̄)
)

@show R_tls
@show t_tls
@show R_gt
@show t_gt

# Errors should be low
@show norm(R_tls - R_gt) * 180 / π
@show norm(t_tls - t_gt)

R_gt1, gt_trans1 = get_groundtruth_Rt(gtruth, t1)
R_gt2, gt_trans2 = get_groundtruth_Rt(gtruth, t2)

# Bring both sets of keypoints into the inertial frame
inertial_pts1 = apply_Rt(matched_pts1, R_gt1', -gt_trans1)
inertial_pts2 = apply_Rt(matched_pts2, R_gt2', -gt_trans2)
show_correspondence!(vis, inertial_pts1, inertial_pts2)

# Bring both sets of keypoints into frame 2
matched_pts1_rotated_tls = apply_Rt(matched_pts1, R_tls, t_tls)
matched_pts1_rotated_gt = apply_Rt(matched_pts1, R_gt, t_gt)
# Visualize with frame 2 3d keypoints; both should match
show_correspondence!(vis, matched_pts2, matched_pts1_rotated_gt, "gt")
show_correspondence!(vis, matched_pts2, matched_pts1_rotated_tls, "tls")
show_correspondence!(vis, matched_pts2, matched_pts1, "invalid")