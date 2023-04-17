# Test for visualizing camera field of view under rototranslation

include("mapping.jl")
include("utils.jl")  # For types

using Rotations
# K from ETH3D dataset example
K = SM3{Float32}([726.301    0.0    356.692;
   0.0    726.301  186.454;
   0.0      0.0      1.0])
# Image pixel index rages
xrange = [0, 458]
yrange = [0, 739]

t = [0; 0; 0.0]  # Starting translation [m]

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

# Go through some preset rototranslation and plot FOV in MeshCat
for i in 1:3:200
    global t = [-cos(0.01*i); -sin(0.01*i); 0.002*i]
    #t = [0; 0; 0.0]
    R = RotXYZ(pi/2, 0.01*i, pi/2 + 0.2 - 0.001*i)
    local T = SM4{Float32}([R t; 0 0 0 1])  # Camera pose in world frame
    plot_fov_polyhedron!(vis, K, inv(T), xrange, yrange)
    sleep(0.1)
end