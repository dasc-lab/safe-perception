function get_cal_params(path)
    f = open(path, "r")
    cal_strs = split(read(f, String))
    fx, fy, cx, cy = parse.(Float64, cal_strs)
    return fx, fy, cx, cy
end

function assemble_K_matrix(fx, fy, cx, cy)
    K = [fx 0 cx;
         0 fy cy;
         0 0 1]
    return K
end
