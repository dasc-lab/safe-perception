# include("PoseEstimation.jl")
# include("utils.jl")
using SafePerception
using SafePerception.PoseEstimation
using BenchmarkTools, Random, Rotations
using LinearAlgebra  # norm
using Printf
PE = PoseEstimation

Random.seed!(42);
# |ϵ| ≤ β; β is an upper bound on the inlier noise.
β = 0.01f0
# Set maximum residual of inliers
# This number should be computed based off β
c̄ = 1
N = 1_000  # Number of points (correspondences)

p1, p2_noisy, R_groundtruth, t_groundtruth, outlier_inds = generate_synthetic_data(N=N)

# Normal least squares
@time R_ls, t_ls = PE.estimate_Rt(p1, p2_noisy;
    method_pairing=PE.Star(),
    β = β,
    method_R=PE.LS(),
    method_t=PE.LS())

@show PE.rotdist(R_ls, R_groundtruth) * 180 / π
@show norm(t_ls - t_groundtruth)

# Truncated least squares (in-place)
@time R_tls, t_tls = PE.estimate_Rt_fast(p1, p2_noisy;
    method_pairing=PE.Star(),
    β = β,
    method_R=PE.TLS(c̄=c̄), # TODO: fix c̄, put in the theoretically correct value based on β
    method_t=PE.TLS(c̄=c̄)
)

@show PE.rotdist(R_tls, R_groundtruth) * 180 / π
@show norm(t_tls - t_groundtruth)

@show norm(R_ls - R_groundtruth)
@show norm(R_tls - R_groundtruth)

# Test inlier detection with scale invariance and max clique
function test_inlier_detection(N, outlier_fraction, complete_frac=1.0)
    @printf("Testing inlier detection with %i correspondences, %1.3f outlier fraction, %1.3f completion fraction\n", N, outlier_fraction, complete_frac)
    p1, p2_noisy, R_groundtruth, t_groundtruth, outlier_inds = generate_synthetic_data(N=N, outlier_fraction=outlier_fraction)
    inlier_inds = [i for i=1:N if !(i in outlier_inds)];
    est_inlier_inds = PE.get_inlier_inds(p1, p2_noisy, β, PE.Complete(complete_frac))
    correct_inlier_count = length(intersect(Set(inlier_inds), Set(est_inlier_inds)))
    est_outlier_inds = [i for i=1:N if !(i in est_inlier_inds)]
    correct_outlier_count = length(intersect(Set(outlier_inds), Set(est_outlier_inds)))

    @printf("Number of predicted inliers: %i\n", length(est_inlier_inds))
    @printf("Number of predicted outliers: %i\n", length(est_outlier_inds))
    @printf("%i out of %i actual inliers predicted correctly\n", correct_inlier_count, length(inlier_inds))
    @printf("%i out of %i actual outliers predicted correctly\n\n", correct_outlier_count, length(outlier_inds))
end

# Test with varying outlier ratios (10% to 99%)
test_inlier_detection(1000, 0.1)
test_inlier_detection(1000, 0.3)
test_inlier_detection(1000, 0.7)
test_inlier_detection(1000, 0.99)

# Test with varying graph completeness
# Any missing edge will split max clique 
test_inlier_detection(100, 0.7, 1)
test_inlier_detection(100, 0.7, 0.99)
test_inlier_detection(100, 0.7, 0.9)
test_inlier_detection(100, 0.7, 0.5)
