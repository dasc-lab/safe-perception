include("PoseEstimation.jl")
using .PoseEstimation
using BenchmarkTools, Random, Rotations
using LinearAlgebra  # norm
using Printf
PE = PoseEstimation

# Simple pre-defined test for inlier rejection
N = 4
p1 = [0 1 0 1;
      0 0 1 1;
      0 0 0 0.0]
R = RotZ(pi/4)
t = [0; 0; 0]
p2 = R*p1 .+ t
# Add inlier noise
p2_noisy = p2
# Add outlier noise
outlier_inds = 2
p2_noisy[:, outlier_inds] += 3*[1; 0; 0]

# Truncated least squares sanity check rotation
R_tls, t_tls = PE.estimate_Rt(p1, p2_noisy;
    method_pairing=PE.Star(),
    method_R=PE.TLS(c̄ = 0.01), 
    method_t=PE.TLS(c̄ = 0.01)
)
@show PE.rotdist(R_tls, R) * 180 / π  # Should be near zero
@show norm(t_tls - t)  # Should be near zero


inlier_inds = [i for i=1:N if !(i in outlier_inds)];
est_inlier_inds = PE.get_inlier_inds(p1, p2_noisy, 0.1, PE.Complete())
correct_inlier_count = length(intersect(Set(inlier_inds), Set(est_inlier_inds)))
@printf("Predicted inliers (expected: [1, 3, 4]):\n")
@show est_inlier_inds
