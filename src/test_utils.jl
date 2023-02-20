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
    # Interpolate to find start and end rotations and translations
    # Columns: timestamp tx ty tz qx qy qz qw
    first = interp_lin(gtruth, time1)
    second = interp_lin(gtruth, time2)
    # TODO(rgg): implement slerp for quaternion interpolation / use Quaternions.jl
    # Extract and normalize quaternions
    xyzw1 = normalize(first[5:8])
    xyzw2 = normalize(second[5:8])
    q1 = PE.Quaternion(xyzw1)
    q2 = PE.Quaternion(xyzw2)
    t1 = first[2:4]
    t2 = second[2:4]
    # Compute relative rotation and convert to rotation matrix
    # Note that (q2*q1')*q1 = q2 => (q2*q1') represents rotation from 1 to 2
    q = PE.quatprod(q2, PE.quatinv(q1))
    R = PE.quat_to_rot(q)
    # Compute relative translation
    t = t2-t1
    return R, t
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