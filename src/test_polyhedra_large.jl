using BenchmarkTools, Logging, Printf
using LinearAlgebra
using Rotations
using DecompUtil
using Polyhedra
using ColorTypes, MeshCat
using BenchmarkTools

# seed from which the decomposition starts
pos1 = [0.5,0.5,0.5] 
pos2 = [-0.5,0,0] 

# Generate a random point cloud with two spheres cut out of it
r1 = 0.7
r2 = 1.5
center1 = [1,0.5,0.5]
center2 = [-1,0,0.]
N = 250000
d = 7  # side length of random cube to generate points in
rand_pts = [d*rand(Float32, 3) .- d/2f0 for i in 1:N]
obs = [p for p in rand_pts if (norm(p-center1) > r1) && (norm(p-center2) > r2)]
@time begin
    # Would be more efficient to pass slices of an array
    obs_x = [ob[1] for ob in obs]
    obs_y = [ob[2] for ob in obs]
    obs_z = [ob[3] for ob in obs]
end
pc = PointCloud(obs)

# defines the bounding box (i.e., -2:2 on x axis, -2:2 on y axis)
bbox = [0.5,0.5,0.5]

# define a dilation radius
dilation_radius = 0.1

# Hyperplanes: point, normal vector (not guaranteed to be collinear?)
#@btime result1 = seedDecomp(pos1, obs, bbox, dilation_radius)
#@btime result2 = seedDecomp(pos2, obs, bbox, dilation_radius)

@time result1 = seedDecomp_3D_fast(pos1, obs_x, obs_y, obs_z, bbox, dilation_radius)
@time result2 = seedDecomp_3D_fast(pos2, obs_x, obs_y, obs_z, bbox, dilation_radius)

# Visualize
visualize = true
if visualize
    if !@isdefined vis
        vis = Visualizer()  # Only need to set up once
    end
    delete!(vis)  # Clear any rendered objects
end

# Construct polyhedron as intersection of half spaces
# H = a'x ≤ a'x_0
ϵ = 0.2  # norm ball error around points
hs1 = [Polyhedra.HalfSpace(r.n, r.n' * r.p) for r in result1]
hs2 = [Polyhedra.HalfSpace(r.n, r.n' * r.p) for r in result2]
p1 = polyhedron(reduce(∩, hs1))
p2 = polyhedron(reduce(∩, hs2))
m1 = Polyhedra.Mesh(p1)
m2 = Polyhedra.Mesh(p2);

# Plot points
if visualize
    solid_green = MeshLambertMaterial(color=RGB(0, 1, 0))
    solid_red = MeshLambertMaterial(color=RGB(1, 0, 0))
    translucent_blue = MeshLambertMaterial(color=RGBA(0, 0, 1, 0.5))
    translucent_purple = MeshLambertMaterial(color=RGBA(0.7, 0, 1, 0.5))
    translucent_purple2 = MeshLambertMaterial(color=RGBA(0.6, 0, 1, 0.5))
    setobject!(vis["seed1"], Sphere(Point3f(pos2), 0.05), solid_red)
    setobject!(vis["seed2"], Sphere(Point3f(pos1), 0.05), solid_red)
    setobject!(vis["pointcloud"], pc, solid_green)
    setobject!(vis["outer1"], m1, translucent_purple)
    setobject!(vis["outer2"], m2, translucent_purple2)
end