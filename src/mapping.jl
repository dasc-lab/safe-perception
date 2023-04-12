# Helper functions for building maps
using Polyhedra
using StaticArrays
include("matching.jl")  # For transformation functions and types, move these / use library?

function generate_fov_halfspaces(K::SM3{Float32},
                                T::SM4{Float32},
                                xrange::Vector{Int},
                                yrange::Vector{Int})
    """
    Generate a list of half-spaces representing the
    boundary of the camera field of view (FOV).
    Used to efficiently generate a safe flight corridor polyhedron.
    Args:
        K: camera matrix
        T: 4x4 rototranslation from world frame to camera frame
        xrange: [min, max] u in image coordinates
        yrange: [min, max] v in image coordinates
    """

    # Construct vectors along "corners" of FOV boundary, in world frame
    upper_left, upper_right, lower_left, lower_right = generate_corner_vectors(K, T, xrange, yrange)
    # Take cross product to get normal vectors
    upper_n = normalize(cross(upper_left, upper_right))
    right_n = normalize(cross(upper_right, lower_right))
    lower_n = normalize(cross(lower_right, lower_left))
    left_n = normalize(cross(lower_left, upper_left))
    fov_normals = [upper_n, right_n, left_n, lower_n]
    static_fov_normals = [SV3{Float64}(n) for n in fov_normals] # For compatibility with other polyhedra
    cam_origin = apply_T([0.f0;0.f0;0.f0], inv(T))   # In world frame
    hs = [HalfSpace(-n, -n'*cam_origin) for n in static_fov_normals]
    return hs
end

function generate_corner_vectors(K::SM3{Float32}, T::SM4{Float32}, xrange, yrange)
    """
    Construct normalized vectors along "corners" of FOV boundary, in world frame rotation.
    Args:
        T: 4x4 rototranslation from world frame to camera frame
    """
    K_inv = inv(K)
    T_inv = inv(T)  # Camera frame to world frame
    R = T_inv[1:3, 1:3]
    t = T_inv[1:3, 4]
    upper_left = R*K_inv*SVector(xrange[1], yrange[2], 1)
    lower_left = R*K_inv*SVector(xrange[2], yrange[2], 1)
    upper_right = R*K_inv*SVector(xrange[1], yrange[1], 1)
    lower_right = R*K_inv*SVector(xrange[2], yrange[1], 1)
    return [normalize(v) for v in (upper_left, upper_right, lower_left, lower_right)]
end


function get_fov_polyhedron(K, T, xrange, yrange)
    """
    Return polyhedron representing the camera field of view.
    This representation does NOT include a maximum depth;
    i.e. it is open in the direction the camera is pointing.
    Args:
        K: camera matrix
        T: 4x4 rototranslation from world frame to camera frame
        xrange: [min, max] u in image coordinates
        yrange: [min, max] v in image coordinates
    """
    hs = generate_fov_halfspaces(K, T, xrange, yrange)
    p = polyhedron(reduce(∩, hs))


    return p
end

function plot_fov_polyhedron!(vis, K, T, xrange, yrange; max_depth=5)
    """
    Plots the camera location and FOV polyhedron in the provided visualizer.
    Args:
        vis: MeshCat visualizer object
        K: camera matrix
        T: 4x4 rototranslation from world frame to camera frame
        xrange: [min, max] u in image coordinates
        yrange: [min, max] v in image coordinates
        max_depth: maximum depth of FOV for visualization purposes
    """
    hs = generate_fov_halfspaces(K, T, xrange, yrange)
    cam_origin = apply_T([0.;0.;0.], inv(T))   # In world frame
    upper_left, upper_right, lower_left, lower_right = generate_corner_vectors(K, T, xrange, yrange)

    # Limit depth arbitrarily (for display only, as it's hard to mesh an unbounded set)
    upper_edge = upper_right - upper_left
    left_edge = lower_left - upper_left
    depth_n = normalize(cross(left_edge, upper_edge))
    max_depth_point = cam_origin + (upper_left.*max_depth)
    push!(hs, HalfSpace(-depth_n, -depth_n'*max_depth_point))
    p = polyhedron(reduce(∩, hs))
    m = Polyhedra.Mesh(p)
    translucent_green = MeshLambertMaterial(color=RGBA(0, 1, 0, 0.5))
    solid_red = MeshLambertMaterial(color=RGB(1, 0, 0))
    yellow = MeshLambertMaterial(color=RGB(1, 1, 0))
    cyan = MeshLambertMaterial(color=RGB(0, 1, 1))
    magenta = MeshLambertMaterial(color=RGB(1, 0, 1))
    white = MeshLambertMaterial(color=RGB(1, 1, 1))
    setobject!(vis["camera"], Sphere(Point3f(cam_origin), 0.05), solid_red)

    # Testing only
    # setobject!(vis["p1"], Sphere(Point3f(cam_origin + upper_left), 0.05), yellow)
    # setobject!(vis["p2"], Sphere(Point3f(cam_origin + upper_right), 0.05), cyan)
    # setobject!(vis["p3"], Sphere(Point3f(cam_origin + lower_left), 0.05), magenta)
    # setobject!(vis["p4"], Sphere(Point3f(cam_origin + lower_right), 0.05), white)
    c5 = MeshLambertMaterial(color=RGB(0.5, 1, 0.5))

    upper_n = -normalize(cross(upper_left, upper_right))
    right_n = -normalize(cross(upper_right, lower_right))
    lower_n = -normalize(cross(lower_right, lower_left))
    left_n = -normalize(cross(lower_left, upper_left))
    # setobject!(vis["n1"], Sphere(Point3f(cam_origin + upper_n), 0.05), yellow)
    # setobject!(vis["n2"], Sphere(Point3f(cam_origin + right_n), 0.05), cyan)
    # setobject!(vis["n3"], Sphere(Point3f(cam_origin + lower_n), 0.05), magenta)
    # setobject!(vis["n4"], Sphere(Point3f(cam_origin + left_n), 0.05), white)
    setobject!(vis["n5"], Sphere(Point3f(cam_origin - depth_n), 0.05), c5)
    setobject!(vis["mdp"], Sphere(Point3f(max_depth_point), 0.05), c5)

    setobject!(vis["fov_bounds"], m, translucent_green)
end

function get_obs_free_polyhedron(points::Matrix{Float32},
                                seed; 
                                T=SM4([I(3) zeros(3); zeros(4)'])::SM4{Float32},
                                ϵ=0,
                                bbox=[5, 5, 5],
                                dilation_radius=0.1)
    """
    Generate polyhedron that includes none of the given points.
    i.e. the safe flight corridor
    Args:
        points: 3xN matrix of xyz points in camera frame
        seed: safe location from which to start propagating (e.g. camera position), in camera frame
        T: 4x4 transformation from world frame to camera frame
        ϵ: radius of norm-ball error around points
        bbox: bounding box for safe polyhedron (should contain field of view in camera frame)
              symmetric +-[x, y, z]
        dilation_radius: parameter for decompUtil
    Returns:
        polyhedron, halfspaces
    """
    # Hyperplanes: point, normal vector 
    N = size(points, 2)
    @info @sprintf("Running decomputil on %i points", N)
    @inbounds result = seedDecomp_3D_fast(seed,
                                          points[1,1:N],
                                          points[2,1:N],
                                          points[3,1:N],
                                          bbox,
                                          dilation_radius)
    # Transform polyhedron to world frame
    T_inv = inv(T)  # Camera frame to world frame
    normals_w = [SV3{Float32}((T_inv * SV4{Float32}([r.n; 1f0]))[1:3]) for r in result]
    points_w = [SV3{Float32}((T_inv * SV4{Float32}([r.p; 1f0]))[1:3]) for r in result]
    result_w = zip(normals_w, points_w)

    hs_shrunk = [Polyhedra.HalfSpace(n, (n' * p) - ϵ) for (n,p) in result_w]
    # hs_shrunk = [Polyhedra.HalfSpace(r.n, r.n' * r.p - ϵ) for r in result]
    p_shrunk = polyhedron(reduce(∩, hs_shrunk))
    return p_shrunk
end