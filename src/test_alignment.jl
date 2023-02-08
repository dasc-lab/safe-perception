include("PoseEstimation.jl")
include("ingest.jl")
include("matching.jl")
using .PoseEstimation
using BenchmarkTools, Random, Rotations
PE = PoseEstimation

"""
Integrated test to visualize alignment of point clouds from
two consecutive frames using transformation from PoseEstimation
"""

# Read in images
path1 = "/root/datasets/training/plant_4/rgb/4043.278005.png"
path2 = "/root/datasets/training/plant_4/rgb/4043.314868.png"
depth_path1 = "/root/datasets/training/plant_4/depth/4043.278005.png"
depth_path2 = "/root/datasets/training/plant_4/depth/4043.314868.png"

img1_color = cv.imread(path1, cv.IMREAD_COLOR) 
img2_color = cv.imread(path2, cv.IMREAD_COLOR)
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)

depth1 = cv.imread(depth_path1, cv.IMREAD_ANYDEPTH) 
depth2 = cv.imread(depth_path2, cv.IMREAD_ANYDEPTH) 
depth1 = pyconvert(Matrix{UInt16}, depth1) ./ 5000  # Divide by 5000 for eth3d dataset
depth2 = pyconvert(Matrix{UInt16}, depth2) ./ 5000

# Read in camera intrinsics matrix
cal_path = "/root/datasets/training/plant_4/calibration.txt"
K = assemble_K_matrix(get_cal_params(cal_path)...)

kp1, kp2, matches = get_matches(img1, img2, "orb")

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

# PoseEstimation expects two 3xN matrices
p1 = np.column_stack([kp.pt for kp in kp1])  
p1 = pyconvert(Matrix{Float64}, p1)
p1_3d = reduce(hcat, get_points_3d(K, p1, depth1))
p1, p1_3d = remove_invalid_returns(p1, p1_3d)

p2 = np.column_stack([kp.pt for kp in kp2])
p2 = pyconvert(Matrix{Float64}, p2)
p2_3d = reduce(hcat, get_points_3d(K, p2, depth2))
p2, p2_3d = remove_invalid_returns(p2, p2_3d)

c̄ = 0.07  # Maximum residual of inliers
@time R_tls, t_tls = PE.estimate_Rt(p1_3d, p2_3d;
    method_pairing=PE.Star(),
    method_R=PE.TLS(c̄), # TODO: fix c̄, put in the theoretically correct value based on β
    method_t=PE.TLS(c̄)
)

# Check against ground truth from ETH3D
