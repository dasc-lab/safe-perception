# Helper functions for evaluating results
function find_interp_idx(arr, val)
    """
    Find the relevant indices for linear interpolation in a 1-D sorted array
    """
    lower = searchsortedlast(arr, val)
    lower = lower < firstindex(arr) ? firstindex(arr) : lower
    upper = searchsortedfirst(arr, val)
    upper = upper > lastindex(arr) ? lastindex(arr) : upper
    return (lower, upper)
end

function interp_lin(arr, t)
    """
    Interpolate between two rows, column-wise using first column to determine interpolation constant.
    """
    # Check degenerate case
    lower, upper = find_interp_idx(arr[:, 1], t)
    lower_row = arr[lower, :]
    if lower == upper
        return lower_row
    end
    diff = arr[upper, :] - lower_row
    λ = (t-lower_row[1]) / diff[1]  # Assume time column monotonically increases
    return lower_row + λ*diff
end

# Check against ground truth from ETH3D
function get_groundtruth_Rt(gtruth, time1, time2)
    """
    Gets relative rototranslation from time1 to time2 from ground truth.
    """
    # Convention for ground truth is R, t go from frame 2 to world frame
    R_1_w, t_1_w = get_groundtruth_Rt(gtruth, time1) 
    R_2_w, t_2_w = get_groundtruth_Rt(gtruth, time2) 
    T_1_w = get_T(R_1_w, t_1_w)
    T_2_w = get_T(R_2_w, t_2_w)

    # take point to w frame from frame 1, then go from w from to frame 2
    T_1_2 = inv(T_2_w) * T_1_w
    return T_1_2[1:3, 1:3], T_1_2[1:3, 4]
end

function extract_R_t(T)
    # Takes in 4x4 rototranslation and return 3x3 R, 3x1 t
    return SM3{Float32}(T[1:3, 1:3]), SV3(T[1:3, 4])
end

function get_groundtruth_Rt(gtruth, time1)
    """
    Gets rototranslation at time1 from ground truth relative to inertial frame.
    """
    # Interpolate to find start and end rotations and translations
    # Columns: timestamp tx ty tz qx qy qz qw
    first = interp_lin(gtruth, time1)
    # TODO(rgg): implement slerp for quaternion interpolation / use Quaternions.jl
    # Extract and normalize quaternions
    xyzw = normalize(first[5:8])
    q1 = PE.Quaternion(xyzw)
    t1 = first[2:4]
    R = PE.quat_to_rot(q1)
    # Compute relative translation
    return R, t1
end

function get_T(R, t)
    return SM4{Float32}([R t; 0 0 0 1])
end

function get_depth(data_folder, dimg_name)
    depth_path = joinpath(data_folder, dimg_name) 
    depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH) 
    depth = pyconvert(Matrix{UInt16}, depth) ./ 5000  # Divide by 5000 for eth3d dataset
    return depth
end

function get_imgs(data_folder, img_name)
    path = joinpath(data_folder, img_name) 
    img_color = cv.imread(path, cv.IMREAD_COLOR) 
    return img_color
end

function get_matched_pts(img1, img2, depth1, depth2)
    """
    Returns two 3xN matrices containing 3D points.
    Columns align to form matched pairs of keypoints.

    Args:
        img1: grayscale image 1 for computing keypoints
        img2: grayscale image 2 for computing keypoints
        depth1: depth image corresponding to image 1
        depth2: depth image corresponding to image 2
    """
    kp1, kp2, matches = get_matches(img1, img2, "orb")
    # PoseEstimation expects two 3xN matrices
    # First, get keypoint lists in 3d
    p1 = np.column_stack([kp.pt for kp in kp1])  
    p1 = pyconvert(Matrix{Float64}, p1)
    p1_3d = get_points_3d(K, p1, depth1)

    p2 = np.column_stack([kp.pt for kp in kp2])
    p2 = pyconvert(Matrix{Float64}, p2)
    p2_3d = get_points_3d(K, p2, depth2)

    # Then, assemble corresponding matrices of points from matches
    n_matches = pyconvert(Int32, py.len(matches))
    matched_pts1 = zeros(Float64, 3, n_matches)
    matched_pts2 = zeros(Float64, 3, n_matches)
    for i in 1:n_matches
        m = matches[i-1]  # Python indexing
        idx1 = pyconvert(Int32, m.queryIdx)+1
        idx2 = pyconvert(Int32, m.trainIdx)+1
        matched_pts1[:, i] = p1_3d[idx1]
        matched_pts2[:, i] = p2_3d[idx2]
    end
    # Finally, clean correspondence list of any pairs that contain either point at the origin (invalid depth)
    matched_pts1, matched_pts2 = remove_invalid_matches(matched_pts1, matched_pts2)
    @info "Matching and cleaning complete"
    return matched_pts1, matched_pts2
end

function generate_synthetic_data(;N=1_000, β=0.01, outlier_fraction=0.5, outlier_noise=40)
    # Generate a ground-truth pose
    R_groundtruth = rand(RotMatrix{3})
    # Generate a ground-truth translation
    t_groundtruth = randn(3)
    # Generate points in frame 1
    p1 = randn(3, N)
    # Generate true p2
    p2 = R_groundtruth * p1  .+ t_groundtruth
    # Make noisy measurements, bounded by inlier noise β
    p2_noisy = copy(p2)
    for i=1:N
        p2_noisy[:, i] += β*(2*rand(3).-1) # Zero-mean uniform noise
    end

    # Add outliers to some percent of data. This noise exceeds inlier noise.
    outlier_inds = [i for i=2:N if rand() < outlier_fraction]
    for i=outlier_inds
        p2_noisy[:, i] += outlier_noise*randn(3)
    end
    return p1, p2_noisy, R_groundtruth, t_groundtruth, outlier_inds
end