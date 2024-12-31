import cv2 
import numpy as np

IMG_PATH = '/Users/tokudataichi/Documents/python_mediapipe/input_images/60left.jpg'
img = cv2.imread(IMG_PATH)

HEIGHT, WIDTH, _ = img.shape
imgCenter = (WIDTH/2.0, HEIGHT/2.0)
pixelPicth = 0.002 # 35判換算　単位mm
baseLine = 1500 # mm
K = np.array([[4437.04178, 0, 2165.73130], [0, 4515.94693, 2783.42064], [0, 0, 1]])  
R = np.array([[0.5, 0, -(np.sqrt(3))/2.0],
              [0, 1, 0],
              [np.sqrt(3)/2.0, 0, 0.5]])
t = np.array([[np.sqrt(3)/2.0*baseLine], [0], [0.5*baseLine]])
print("K:", K)
print("R:",R)
print("t:",t)

Pleft = np.hstack((K, np.zeros((3, 1))))
Pright = np.hstack((K @ R, K @ t))
print("Pleft:",Pleft)
print("Pright:",Pright)

leftPoints = np.array(imgCenter)
rightPoints = np.array(imgCenter)

points4D = cv2.triangulatePoints(Pleft, Pright, leftPoints, rightPoints)
points3D = points4D[:3] / points4D[3] # 3行n列
print("points3D:", points3D)

pointsGT = np.array([0, 0, 1500])
