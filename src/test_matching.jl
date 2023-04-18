"""
Test that visualizes feature-based correspondences between two RGB frames.
Will save 2D images and show a 3D visualization of the matches (projected using depth image).
"""

include("ingest.jl")
include("matching.jl")
using GeometryBasics, ColorTypes, CoordinateTransformations, MeshCat, LinearAlgebra
using DelimitedFiles
using Logging, Printf
debug_logger = Logging.ConsoleLogger(Logging.Info)

# NOTE: test requires downloading an ETH3D dataset.
# Tested with plant_4 and sofa_2
# To download, run:
# python3 download_eth3d_slam_datasets.py
include("setup_test.jl")

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

function test_match(img1, img2, feature_string::String, use_flann=false)
    """
    Find corresponding points across two frames using a feature detector.
    Visualize the matches in 2D and save to file.
    """
    kp1, kp2, matches = get_matches(img1,
                                    img2,
                                    feature_string,
                                    use_flann=use_flann)
    with_logger(debug_logger) do
        @info @sprintf("Keypoints in frame 1 using %s: %s", feature_string, np.shape(kp1)[0])
        @info @sprintf("Keypoints in frame 2 using %s: %s", feature_string, np.shape(kp2)[0])
        @info @sprintf("Matches using %s: %s", feature_string, np.shape(matches)[0])
    end
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[pyslice(60)],py.None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("/root/src/" * feature_string * "_matches.jpg", img3)
end

test_match(img1, img2, "orb", true)
test_match(img1, img2, "sift")
test_match(img1, img2, "akaze")
# test_match(img1, img2, "surf")  # Only if SURF is installed

# Test 3D correspondences
kp1, kp2, matches = get_matches(img1, img2, "orb")

# Get point cloud by reprojecting + depth info
# Invalid returns will be represented as zeros (projected to origin)
# Assemple in numpy and convert to Julia. Not sure if optimal.
kpoints1 = np.column_stack([kp.pt for kp in kp1]) 
kpoints1 = pyconvert(Matrix{Float64}, kpoints1)

kpoints2 = np.column_stack([kp.pt for kp in kp2])
kpoints2 = pyconvert(Matrix{Float64}, kpoints2)

# Get 3D points for each keypoint in each frame
points1_3d = get_points_3d(K, kpoints1, depth1)
points2_3d = get_points_3d(K, kpoints2, depth2)
# Visualize correspondences
for m in matches
    show_correspondence!(vis, m, kpoints1, kpoints2, img1_color, img2_color, points1_3d, points2_3d)
end
# Visualize entire point cloud in frame 1
show_pointcloud_color!(vis, depth1, convert_py_rgb_img(img1_color), K)