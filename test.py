import cv2 
import numpy as np
import matplotlib.pyplot as plt

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

def plot_3Dskeleton(landmarks, connections, matchlist):
    fig = plt.figure(figsize = (8, 8))
    ax= fig.add_subplot(111, projection='3d')
    ax.scatter(landmarks[:, 0], landmarks[:,1],landmarks[:,2], s = 1, c = "blue")
    # for i in range(len(landmarks[:, 0])):
    #     ax.text(landmarks[i,0], landmarks[i,1], landmarks[i,2], str(i), dir=None, color="black", fontsize=8, ha='right', va='bottom')
    set_equal_aspect(ax)
    # 骨格情報からボーンを形成
    for connection in connections:
        start_joint, end_joint = connection
        if matchlist[start_joint] == 1 and matchlist[end_joint] == 1:
            x = [landmarks[start_joint, 0], landmarks[end_joint, 0]]
            y = [landmarks[start_joint, 1], landmarks[end_joint, 1]]
            z = [landmarks[start_joint, 2], landmarks[end_joint, 2]]
            plt.plot(x, y, z, c='red', linewidth=1)
            

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.autoscale(None)
    plt.title("3D Skeleton Visualization")
    plt.show()

def set_equal_aspect(ax):
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = abs(limits[:, 1] - limits[:, 0])
    centers = np.mean(limits, axis=1)
    max_span = max(spans)
    new_limits = np.array([
        centers - max_span / 2,
        centers + max_span / 2
    ]).T
    ax.set_xlim3d(new_limits[0])
    ax.set_ylim3d(new_limits[1])
    ax.set_zlim3d(new_limits[2])

leftPoints = np.array(imgCenter)
rightPoints = np.array(imgCenter)

points4D = cv2.triangulatePoints(Pleft, Pright, leftPoints, rightPoints)
points3D = points4D[:3] / points4D[3] # 3行n列
print("points3D:", points3D)
leftCamPos = np.zeros((3, 1))
rightCamPos = np.zeros((3, 1))+t
pointsGT = np.array([0, 0, 1500])
