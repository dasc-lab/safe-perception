# make sure it uses the system-instaled python backend
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"
ENV["JULIA_PYTHONCALL_EXE"] = "/usr/bin/python3"

using PythonCall

# import opencv

cv = pyimport("cv2")

# run the sift commands

img = cv.imread("frb.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()

kp = sift.detect(gray, pybuiltins.None)

img = cv.drawKeypoints(gray, kp, img)

cv.imwrite("sift_keypoints_julia.jpg", img)
