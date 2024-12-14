import numpy as np
import cv2 as cv
import glob
import pandas as pd

# file path 
file_path = '/Users/tokudataichi/Documents/python_mediapipe/calibration_images/*.JPG'

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# chessboard grid points

cbrow = 7
cbcol = 10


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((cbrow * cbcol,3), np.float32)
objp[:,:2] = np.mgrid[0:cbcol,0:cbrow].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(file_path)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray,(cbcol,cbrow),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
      objpoints.append(objp)
      corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
      imgpoints.append(corners2)

      # Draw and display the corners
      cv.drawChessboardCorners(img, (cbcol,cbrow), corners2,ret)
      cv.imshow("img",img)
      cv.waitKey(500)
    else:
      print("見つけられませんでした"+ fname)

cv.destroyAllWindows()

# カメラキャリブレーション
retval,mtx,dist,rvecs,tvecs = cv.calibrateCamera(objpoints,imgpoints,gray.shape[::-1],None,None)

print("再投影誤差: " , retval)
print("カメラ行列:", "\n", mtx)
print("歪み係数:", "\n", dist)

rvecs_list = [vec.flatten() for vec in rvecs]
tvecs_list = [vec.flatten() for vec in tvecs]
df_r = pd.DataFrame(rvecs_list, columns=['X', 'Y', 'Z'])
df_t = pd.DataFrame(tvecs_list, columns=['X', 'Y', 'Z'])

print("回転ベクトル:")
print(df_r)
print("並進ベクトル:")
print(df_t)

for i, fname in enumerate(images):
  img = cv.imread(fname)
  h,  w = img.shape[:2]
  newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
  print("新カメラ行列: " , "\n", newcameramtx)

  # # 歪補正
  # dst = cv.undistort(img, mtx, dist, None, newcameramtx)

  # # 画像の切り落とし
  # x,y,w,h = roi
  # dst = dst[y:y+h, x:x+w]
  # cv.imshow(dst)

