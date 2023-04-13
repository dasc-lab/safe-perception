# Make sure we use the system-installed Python backend
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/usr/bin/python3"

using PythonCall
cv = pyimport("cv2")

function get_cal_params(path)
    """
    Gets the camera calibration parameters from the calibration.txt file.
    For use with ETH3D dataset.
    """
    f = open(path, "r")
    cal_strs = split(read(f, String))
    fx, fy, cx, cy = parse.(Float64, cal_strs)
    return fx, fy, cx, cy
end

function assemble_K_matrix(fx, fy, cx, cy)
    """
    Assembles the camera intrinsics matrix from the calibration parameters.
    """
    K = [fx 0 cx;
         0 fy cy;
         0 0 1]
    return K
end

function get_depth(data_folder, dimg_name)::Matrix{Float32}
    """
    Gets a depth image from the data folder.
    Args:
        data_folder: path to folder containing images
        dimg_name: name of depth image file (e.g. "depth_0.png")
    Returns: a Julia array depth image in meters.
    """
    depth_path = joinpath(data_folder, dimg_name) 
    # TODO(rgg): just read this directly into Julia
    depth = cv.imread(depth_path, cv.IMREAD_ANYDEPTH) 
    depth = pyconvert(Matrix{UInt16}, depth) ./ 5000  # Divide by 5000 for eth3d dataset
    return depth
end

function get_imgs(data_folder::String, img_name::String)::Py
    """
    Gets an image from the data folder.
    Args:
        data_folder: path to folder containing images
        img_name: name of image file (e.g. "image_0.png")
    Returns: an OpenCV image in BGR format.
    """
    path = joinpath(data_folder, img_name) 
    img_color = cv.imread(path, cv.IMREAD_COLOR) 
    return img_color
end