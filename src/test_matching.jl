include("ingest.jl")
include("matching.jl")
using GeometryBasics, ColorTypes, CoordinateTransformations, MeshCat, LinearAlgebra
using DelimitedFiles
using Logging, Printf
debug_logger = Logging.ConsoleLogger(Logging.Info)

if !@isdefined vis
    vis = Visualizer()  # Only need to set up once
end
delete!(vis)  # Clear any rendered objects

# NOTE: test requires downloading "plant_4" eth3d dataset.
# To download, run:
# python3 download_eth3d_slam_datasets.py

function test_match(img1, img2, feature_string::String)
    kp1, kp2, matches = get_matches(img1, img2, feature_string)
    with_logger(debug_logger) do
        @info @sprintf("Keypoints in frame 1 using %s: %s", feature_string, np.shape(kp1)[0])
        @info @sprintf("Keypoints in frame 2 using %s: %s", feature_string, np.shape(kp2)[0])
        @info @sprintf("Matches using %s: %s", feature_string, np.shape(matches)[0])
    end
    img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[pyslice(20)],py.None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imwrite("/root/src/" * feature_string * "_matches.jpg", img3)
end

# Read in images
# Tested with plant_4 and sofa_2
df = joinpath("/root/datasets/training/plant_4/")
img_filenames = [joinpath("rgb", f) for f in readdir(joinpath(df, "rgb"))]
depth_filenames = readdlm(joinpath(df, "depth.txt"))[:, 2];
file_ind1 = 12
file_ind2 = 13
path1 = joinpath(df, img_filenames[file_ind1])
path2 = joinpath(df, img_filenames[file_ind2])
depth_path1 = joinpath(df, depth_filenames[file_ind1])
depth_path2 = joinpath(df, depth_filenames[file_ind2])

img1_color = cv.imread(path1, cv.IMREAD_COLOR) 
img2_color = cv.imread(path2, cv.IMREAD_COLOR)
img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY)
depth1 = cv.imread(depth_path1, cv.IMREAD_ANYDEPTH) 
depth2 = cv.imread(depth_path2, cv.IMREAD_ANYDEPTH) 

# Read in camera intrinsics matrix
cal_path = joinpath(df, "calibration.txt")
K = assemble_K_matrix(get_cal_params(cal_path)...)

test_match(img1, img2, "orb")
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

# Can we just read these in as Julia matrices?
depth1 = pyconvert(Matrix{UInt16}, depth1) ./ 5000  # Divide by 5000 for eth3d dataset
depth2 = pyconvert(Matrix{UInt16}, depth2) ./ 5000

# Get 3D points for each keypoint in each frame
points1_3d = get_points_3d(K, kpoints1, depth1)
points2_3d = get_points_3d(K, kpoints2, depth2)
# Visualize correspondences
for m in matches
    show_correspondence!(vis, m, kpoints1, kpoints2, img1_color, img2_color, points1_3d, points2_3d)
end
# Visualize entire point cloud in frame 1
show_pointcloud_color!(vis, depth1, img1_color, K)