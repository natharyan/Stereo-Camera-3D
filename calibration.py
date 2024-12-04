import numpy as np
import cv2 as cv
import os
import glob
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((10*7,3), np.float32)
objp[:,:2] = np.mgrid[0:10,0:7].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

img_list = os.listdir('datasets/Camera Calibration(Samsung M02)')
img_list = [i for i in img_list if i.endswith('.jpg')]
# img_list.remove('20241116_210603.jpg')
for fname in img_list:
    print(fname)
    img = cv.imread('datasets/Camera Calibration(Samsung M02)/' + fname,cv.COLOR_BGR2GRAY)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (10,7), None)
 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        print('imgpoints:', imgpoints)
        cv.drawChessboardCorners(img, (10,7), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)
    else:
        print("no img points found")
 
cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savez('intrinsics/shreyansh_calibration_data.npz', mtx=mtx, dist=dist)

# distortion correction
# img = cv.imread('datasets/Camera Calibration(Samsung S20 fe)/20241116_210603.jpg')
# h,  w = img.shape[:2]
# newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# # undistort
# dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]

# cv.imwrite('calibresult.png', dst)

#TODO: change opencv code parts