using BenchmarkTools, Logging, Printf
using Rotations
using DecompUtil
using ColorTypes, MeshCat

# seed from which the decomposition starts
pos = [0.3,0.3,0.3] 

# define all the obstacle points (cube)
obs = [0 0 0 0 1 1 1 1;
       0 0 1 1 0 0 1 1;
       0 1 0 1 0 1 0 1.0]
obs = [Vector(c) for c in eachcol(obs)]

# defines the bounding box (i.e., -2:2 on x axis, -2:2 on y axis)
bbox = [2,2,2.]

# define a dilation radius
dilation_radius = 0.1

# Hyperplanes: point, normal vector (not guaranteed to be collinear?)
result = seedDecomp(pos, obs, bbox, dilation_radius)

# Visualize
if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

# Construct polyhedron as intersection of half spaces
# H = a'x ≤ a'x_0
ϵ = 0.2  # norm ball error around points
hs = [Polyhedra.HalfSpace(r.n, r.n' * r.p) for r in result]
hs_shrunk = [Polyhedra.HalfSpace(r.n, r.n' * r.p - ϵ) for r in result]
p = polyhedron(reduce(∩, hs))
p_shrunk = polyhedron(reduce(∩, hs_shrunk))
m = Polyhedra.Mesh(p)
m_shrunk = Polyhedra.Mesh(p_shrunk)

# Plot points
solid_green = MeshLambertMaterial(color=RGB(0, 1, 0))
solid_red = MeshLambertMaterial(color=RGB(1, 0, 0))
translucent_blue = MeshLambertMaterial(color=RGBA(0, 0, 1, 0.5))
translucent_purple = MeshLambertMaterial(color=RGBA(0.7, 0, 1, 0.5))
setobject!(vis["seed"], Sphere(Point3f(pos), 0.05), solid_red)
for (i, c) in enumerate(obs)
    pt1 = Point3f(c)
    setobject!(vis["pc1_"*string(i)][string(i)], Sphere(pt1, 0.05), solid_green)
end
setobject!(vis["shrunk"], m_shrunk, translucent_blue)
setobject!(vis["outer"], m, translucent_purple)