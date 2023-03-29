module PoseEstimation

# https://github.com/dev10110/GraduatedNonConvexity.jl
using LinearAlgebra, StaticArrays, GraduatedNonConvexity, Parameters
# https://github.com/dev10110/ParallelMaximumClique.jl
using Graphs, ParallelMaximumClique # For max clique. See also SimpleWeightedGraph, but not compatible with ParallelMaximumClique
using Random, Rotations 
using Printf, Logging
debug_logger = Logging.ConsoleLogger(Logging.Info)

SV3{F} = SVector{3,F}
SV4{F} = SVector{4,F}
Quaternion{F} = SVector{4,F}  # x y z w

function rotdist(R1, R2)
    s = (tr(R1 * R2') - 1) / 2 
    s = clamp(s, -1.0,1.0) # to correct for potential float pt errors
    return acos(s)
end

function ransac(N, data, solver, res, c, n; verbose=false, min_inlier_ratio=0.1, max_iterations=1000, initial_guess=solver(ones(N), data))
    iter = 0
    bestErr = Inf
    x_best = copy(initial_guess)
    rs = res(x_best, data)
    best_error = mapreduce(r->max(r^2, c^2),  +, rs)
    #@show best_error

    for iter = 1:max_iterations
        
        # sample randomly
        sample = rand(1:N, n)
        sample_w = zeros(N)
        sample_w[sample] .= 1
        # solve for the best solution on these points
        sample_x = solver(sample_w, data)
        
        # check for inliers
        rs = res(sample_x, data)
        inliers = [i for i=1:N if abs(rs[i]) <= c]
        #@show length(inliers)
        if length(inliers) > min_inlier_ratio * N
            #@show length(inliers)
            # fit to all inliers
            inlier_w = zeros(N)
            inlier_w[inliers] .= 1
            inlier_x = solver(inlier_w, data)
            inlier_rs = res(inlier_x, data)
            inlier_error = mapreduce(r-> max(r^2, c^2), +, inlier_rs[inliers])
            #@show inlier_error, best_error
            if inlier_error < best_error
                #@show "replacing"
                x_best = inlier_x
                best_error = inlier_error
            end
        end
    end
    return x_best
end

# returns qa ∘ qb
function quatprod(qa, qb)
    return Ω1(qa) * qb
end

# returns inv(q)
function quatinv(q::Quaternion)
    return Quaternion(-q[1], -q[2], -q[3], q[4])
end

# function vechat(q)
#     if length(q) == 3
#         return q[1], q[2], q[3], zero(q[3])
#     end
#     if length(q) == 4
#         return q[1], q[2], q[3], q[4]
#     end
# end

# function vechat(q::SV3{F}) where {F}
#     return q[1], q[2], q[3], zero(F)
# end

# function vechat(q::SV4{F}) where {F}
#     return q[1], q[2], q[3], q[4]
# end


# returns operator Ω1(q) such that q ∘ v = Ω1(q) v̂
function Ω1(q::Quaternion)# left mul operator

    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    q4 = q[4]
    
    return @SMatrix [
        [ q4 ;; -q3 ;;  q2 ;; q1] ;
        [ q3 ;;  q4 ;; -q1 ;; q2] ;
        [-q2 ;;  q1 ;;  q4 ;; q3] ;
        [-q1 ;; -q2 ;; -q3 ;; q4]
    ]
    
end
function Ω1(q::SV3)
    return Ω1(Quaternion(q..., 0))
end

# returns operator Ω2(q) such that v ∘ q = Ω2(q) v̂
function Ω2(q::Quaternion) # right mul operator
    
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]
    q4 = q[4]
    
    return @SMatrix [
        [ q4 ;;  q3 ;; -q2 ;; q1] ;
        [-q3 ;;  q4 ;;  q1 ;; q2 ] ;
        [ q2 ;; -q1 ;;  q4 ;; q3 ] ;
        [-q1 ;; -q2 ;; -q3 ;; q4]
    ]
    
end

function Ω2(q::SV3)
    return Ω2(Quaternion(q..., 0))
end

function quat_to_rot(q::Quaternion{F}) where {F}
    
    R̂ = Ω1(q) * Ω2(quatinv(q))
    
    R = R̂[1:3, 1:3]
    
    return SMatrix{3,3,F,9}(R)
end

function construct_Q(N, a::AF, b::AF, w::VF) where {F, AF<:AbstractArray{F}, VF<:AbstractVector{F}}
    # construct Q matrix
    Q = zero(MMatrix{4,4, F, 16})

    @inbounds for i=1:N
        if w[i] != 0
            qa = Quaternion(a[1, i], a[2, i], a[3, i], zero(F))
            qb = Quaternion(b[1, i], b[2, i], b[3, i], zero(F))
            Ω1b = Ω1(qb)
            Ω2a = Ω2(qa)
            Qi = Ω1b' * Ω2a
            wQiQi = w[i] * Qi + Qi'
            Q .-= wQiQi
        end
    end
    return Q
end

"""
    estimate_R(a, b, w)

solves the problem

R^* = min sum_{i=1}^N w_i ||b_i - R a_i||^2

in closed form
"""
function _estimate_R(a::AF, b::AF, w=ones(F, size(a, 2))) where {F, AF <: AbstractArray{F}}
    """
    method: cast the problem in quaternions
    write the problem as min q^T Q q such that ||q|| = 1
    notice the problem is an eigenvector problem
    return eigenvector with smallest eigenvalue
    """
    
    @assert size(a) == size(b)
    N = size(a, 2)
    @assert length(w) == N
    
    Q = construct_Q(N, a, b, w)
    
    # get min eigenvector 
    q = eigvecs(Q)[:,1] |> real |> Quaternion{F} 

    # convert to rotation matrix
    R = quat_to_rot(q)
    
    return R
    
end

"""
    estimate_t(s, w)

solves the problem

R^* = min sum_{i=1}^N w_i ||s_i - t||^2

in closed form

"""
function _estimate_t(s, w=ones(size(s,2)))
    
    D, N = size(s) # t \in R^D, and there are N points to match
    
    P = sum(w)*I(D)
    q = sum(w[i] * s[:, i] for i in 1:N)
    
    return P \ q
end

abstract type PairingMethod end

@with_kw struct Star <:PairingMethod
    n::Integer = 1
end

@with_kw struct Complete <:PairingMethod
    frac::Float64 = 1.0
end

function make_pairs(m::Star, N)
    is = ones(Int, N-1) * m.n
    js = [j for j=1:N if j != m.n]
    return is, js
end

# TODO(rgg): add other sparse topologies?

function make_pairs(m::Complete, N::T) where {T}
    
    is = T[]
    js = T[]
    for i=1:N, j=1:N
        if i > j && (m.frac == 1 || rand() < m.frac)
            push!(is, i)
            push!(js, j)
        end
    end
    
    return is, js
end

abstract type LsqMethod end

struct LS <: LsqMethod
end

@with_kw struct GM <: LsqMethod
    c̄
    max_iterations = 1000
    μ_factor = 1.4
    verbose = false
    rtol = 1e-6
end
GM(c, kwargs...) = GM(c̄=c; kwargs...)

@with_kw struct TLS <: LsqMethod
    c̄
    max_iterations = 1000
    μ_factor = 1.4
    rtol = 1e-6
    verbose = false
end
TLS(c) = TLS(c̄ = c)


function wls_solver_R(w, data)
    a, b = data
    return _estimate_R(a, b, w)
end

function residuals_R(R, data)
    a, b = data
    N = size(a, 2)
    δ = b - R*a
    return [norm(δ[:, i]) for i=1:N]
end

function wls_solver_t(w, s)
    return _estimate_t(s, w)
end

function residuals_t(t, s)
    return [norm(s[:, i] - t) for i=1:size(s, 2)]
end


function estimate_R(method::LS, a, b)
    R = _estimate_R(a, b)
    return R
end

function estimate_R(method::TLS, a, b)
    N = size(a, 2)
    data = (a, b)
    R = GNC_TLS(N, data, wls_solver_R, residuals_R, method.c̄; 
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose=method.verbose,
        rtol = method.rtol
    )
    return R
end

function estimate_R(method::GM, a, b)
    N = size(a, 2)
    data = (a, b)
    R = GNC_GM(N, data, wls_solver_R, residuals_R, method.c̄; 
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose=method.verbose,
        rtol = method.rtol
    )
    return R
end

function estimate_t(method::LS, p1,p2,R)
    s = p2 - R*p1
    t = _estimate_t(s)
    return t
end

function estimate_t(method::TLS, p1,p2,R)

    s = p2 - R * p1
    N = size(s, 2)
    
    R = GNC_TLS(N, s, wls_solver_t, residuals_t, method.c̄; 
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose=method.verbose,
        rtol = method.rtol
    )
    return R
end

function estimate_t(method::GM, p1, p2, R)
    s = p2 - R * p1
    N = size(s, 2)
    
    R = GNC_TLS(N, s, wls_solver_t, residuals_t, method.c̄; 
        max_iterations = method.max_iterations,
        μ_factor = method.μ_factor,
        verbose=method.verbose,
        rtol = method.rtol
    )
    return R
end

function estimate_Rt(p1, p2; method_pairing::PairingMethod, method_R::LsqMethod, method_t::LsqMethod)
    
    N = size(p1, 2)
    
    # In order to estimate rototranslation, we need Translation Invariant Measurements (TIMs) 
    # across the two frames. This allows for outlier rejection.
    # Pairing method is configurable and not directly related to the keypoints themselves.
    is, js = make_pairs(method_pairing, N)
    
    a = p1[:, is] - p1[:, js]
    b = p2[:, is] - p2[:, js]
    
    R = estimate_R(method_R, a, b)
    
    t = estimate_t(method_t, p1, p2, R)
    
    return R, t
    
end

function get_inlier_inds(p1, p2, ϵ, method_pairing::PairingMethod)
    """
    Use translation invariant measurements to find inlier indices using max-clique inlier selection.
    Returns: list of indices 
    Args:
        p1: 3xN list of points in frame 1
        p2: 3xN list of points in frame 2 that correspond column-wise to points in p1.
            May contain false correspondences, and noise determined by sensing & feature detection.
        ϵ: maximum noise for inlier correspondences
        method_pairing: pairing method to create TIMs from keypoints
    Assumes scaling is unity. TODO(rgg): implement scale estimation?
    Optimal (most accurate) pairing method is to form a complete graph, but this is slow.
    """
    N = size(p1, 2)
    # Create TIMs (see: TEASER paper)
    is, js = make_pairs(method_pairing, N)  # TODO(rgg): use Graphs.jl throughout?
    E = length(is)
    @info @sprintf("Number of edges in TIM graph: %i\n", E)
    # Vectors from keypoints in frame 1 to other keypoints in frame 1
    # Columns in ̄a correspond to columns in ̄b
    tim_̄a = p1[:, is] - p1[:, js] 
    tim_̄b = p2[:, is] - p2[:, js]
    # Create TRIMs
    G = SimpleGraph(N)  # Unweighted edges for compatibility with max clique library
    # Skip edges that are not consistent with estimated scale (assume s=1 for now)
    # s_ij = s + o_ij^s + ϵ_ij^s; TRIM is equal to true scaling + outlier noise + modeled noise 
    # relative noise |ϵ_ij| ≤ 2*|ϵ| as the ϵ is the noise for measurements in each frame
    ϵ_ij = 2*ϵ
    ϵ_ij_s = ϵ_ij ./ norm.(eachcol(tim_̄a))
    s_expected = 1  # Assumption (static environment, camera parameters)
    # Compute scale estimate for each TRIM (edge weights in graph)
    s = norm.(eachcol(tim_̄b)) ./ norm.(eachcol(tim_̄a))
    # Construct graph from TRIMs
    for k in 1:E
        # All TRIMs with scale outside of s ± ϵ_ij_s are inconsistent, do not add these edges
        @debug @sprintf("Processing edge %i-%i with s=%f and scale bounds +/-%f\n", is[k], js[k], s[k], ϵ_ij_s[k])
        if s_expected-ϵ_ij_s[k] <= s[k] <= s_expected+ϵ_ij_s[k]
            add_edge!(G, is[k], js[k])
        end
    end
    @info @sprintf("Number of edges in pruned TRIM graph: %i\n", length(edges(G)))

    # For debugging only, not needed when using with max clique
    # scale_consistent_inds = Set()
    # for e in edges(G)
    #     push!(scale_consistent_inds, e.src)
    #     push!(scale_consistent_inds, e.dst)
    # end
    # return scale_consistent_inds

    # Find maximum clique in remaining graph to get inliners
    clique = maximum_clique(G)

    # Return indices of vertices (points) that belong to inlier TRIMs.
    return clique
end

σmin(A) = sqrt(max(0, eigmin(A'*A)))


# note, this function scales as N^4
# only pass in inliers to p1, i.e. pass in p1[:, inliner_idx]
# this function implements different maths than in Teaser
# since the proof in Teaser is wrong
# a (probably) good anytime approximation will be to run this function with random sets of 4 inds
# and take the lowest bound produced. This approximation will still be an upper-bound to the error
function ϵR(p1, β)
    
    N = size(p1, 2)
    
    best_bound = Inf

    @views @inbounds for i=1:N
        
        PI = SMatrix{3,3}(p1[:, [i,i,i]])
        
        for j=(i+1):N, h=(j+1):N, k=(h+1):N
        
           U = SMatrix{3,3}(p1[:, [j,h,k]]) - PI

           σ = σmin(U)
            
           if σ > 0
                bound = 2*sqrt(3)*(2β) / σ
                best_bound = min(bound, best_bound)
           end
            
        end
    end
    
    return best_bound
    
end

function ϵt(β)
    return (9 + 3*sqrt(3))*β
end


end