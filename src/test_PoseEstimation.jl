include("PoseEstimation.jl")
using .PoseEstimation
using BenchmarkTools, Random, Rotations
using LinearAlgebra  # norm
PE = PoseEstimation

Random.seed!(42);

## construct ground truth data
N = 1_000 # number of correspondences
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
outlier_inds = [i for i=2:N if rand() < 0.70]
for i=outlier_inds
    p2_noisy[:, i] += 3*randn(3)
end

# set maximum residual of inliers
# this number should be computed based off β
c̄ = 0.005

# Normal least squares
@time R_ls, t_ls = PE.estimate_Rt(p1, p2_noisy;
    method_pairing=PE.Star(),
    method_R=PE.LS(),
    method_t=PE.LS())

@show PE.rotdist(R_ls, R_groundtruth) * 180 / π
@show norm(t_ls - t_groundtruth)

# Truncated least squares
@time R_tls, t_tls = PE.estimate_Rt(p1, p2_noisy;
    method_pairing=PE.Star(),
    method_R=PE.TLS(c̄ = 0.07), # TODO: fix c̄, put in the theoretically correct value based on β
    method_t=PE.TLS(c̄ = 0.07)
)

@show PE.rotdist(R_tls, R_groundtruth) * 180 / π
@show norm(t_tls - t_groundtruth)

# estimate the error bound for rotation matrices
# this is a bound on the maximum ||R - R̂||_F
inlier_inds = [i for i=1:N if !(i in outlier_inds)];
@time begin
    bb = Inf
    for i=1:1000
        # Randomly select subset of inliers
        global bb = min(bb, PE.ϵR(p1[:, rand(inlier_inds, 4)], β))
    end
end
@show bb
@show PE.ϵt(β)

@show norm(R_ls - R_groundtruth)
@show norm(R_tls - R_groundtruth)

# Test inlier detection with scale invariance and max clique
est_inlier_inds = PE.get_inlier_inds(p1, p2_noisy, β, PE.Star())
correct_inlier_count = length(intersect(Set(inlier_inds), Set(est_inlier_inds)))
est_outlier_inds = [i for i=1:N if !(i in est_inlier_inds)]
correct_outlier_count = length(intersect(Set(outlier_inds), Set(est_outlier_inds)))
@show length(outlier_inds)
@show correct_outlier_count
@show length(inlier_inds)
@show correct_inlier_count
