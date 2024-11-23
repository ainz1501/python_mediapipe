import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 骨格情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# 画像を読み込む サイズは4284 × 5712
image_path = '/Users/tokudataichi/Documents/python_mediapipe/left0_image.JPG'
image = cv2.imread(image_path)
image2_path = '/Users/tokudataichi/Documents/python_mediapipe/right50_image.JPG'
image2 = cv2.imread(image2_path)

# BGRをRGBに変換（MediaPipeがRGBを期待するため）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# カメラ座標版キーポイント構造体
class cam_landmark:
    x:float
    y:float

    def __init__(self, input_x, input_y) -> None:
        self.x = input_x
        self.y = input_y

# 平行ステレオビジョン
def stereo_vision_easy(pose1, pose2, left1, left2, right1, right2):
    # holistic_3d_landmarks =[46][3]
    pose_3d_landmarks = []
    right_hand_3d = []
    left_hand_3d = []
    baseline = 500.0 # 単位は[mm]
    forcus = 26.0 # 単位は[mm]
    pixel_pitch = 0.0002 #　単位は[mm]

    if pose1 and pose2:
        for p1, p2 in zip(pose1, pose2):
            joint_x = (baseline*p1.x)/(p1.x-p2.x)
            joint_y = (baseline*p1.y)/(p1.x-p2.x)
            joint_z = (forcus*p1.x)/(pixel_pitch*(p1.x-p2.x))

            print(f"x:{joint_x}, y:{joint_y}, z:{joint_z}")
            pose_3d_landmarks.append([joint_x, joint_y, joint_z])
        pose_3d_landmarks = np.array(pose_3d_landmarks)
    
    fig = plt.figure(figsize = (8, 8))
    ax= fig.add_subplot(111, projection='3d')
    ax.scatter(pose_3d_landmarks[:, 0], pose_3d_landmarks[:,1],pose_3d_landmarks[:,2], s = 1, c = "blue")

    for connection in POSE_CONNECTIONS:
        start_joint, end_joint = connection
        x = [pose_3d_landmarks[start_joint, 0], pose_3d_landmarks[end_joint, 0]]
        y = [pose_3d_landmarks[start_joint, 1], pose_3d_landmarks[end_joint, 1]]
        z = [pose_3d_landmarks[start_joint, 2], pose_3d_landmarks[end_joint, 2]]
        plt.plot(x, y, z, c='red', linewidth=1)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Skeleton Visualization")
    plt.legend()
    plt.show()

# カメラ座標への変換（同一デバイス使用前提）
def transform2camera(result, width, height):
    cam_landmarks = [] # カメラ座標を記録する配列を生成
    for index, landmark in enumerate(result.landmark):
        # キーポイントの値をカメラ座標系（画像中央が原点）に変換
        cam_x = landmark.x * width - width/2.0
        cam_y = landmark.y * height - height/2.0
        # cam_landmarkのインスタンスの生成と同時に、x座標、y座標の値を入力
        cam_landmarks.append(cam_landmark(cam_x, cam_y)) 

        # print(f"cam_landmarks[{index}].x: {cam_landmarks[index].x}")

    return cam_landmarks
        
def transform_result(results, image):
    height, width, _ = image.shape

    if results.pose_landmarks:
        pose_cam_landmarks = transform2camera(results.pose_landmarks, width, height)
        # for index, cam_landmarks in enumerate(pose_cam_landmarks):
        #     print(f"pose_cam_landmarks[{index}].x : {cam_landmarks.x}")
    else:
        pose_cam_landmarks = None
    if results.left_hand_landmarks:
        left_hand_cam_landmarks = transform2camera(results.left_hand_landmarks, width, height)
        # for index, cam_landmarks in enumerate(left_hand_cam_landmarks):
        #     print(f"left_hand_cam_landmarks[{index}].x : {cam_landmarks.x}")
    else:
        left_hand_cam_landmarks = None
    if results.right_hand_landmarks:
        right_hand_cam_landmarks = transform2camera(results.right_hand_landmarks, width, height)
        # for index, cam_landmarks in enumerate(right_hand_cam_landmarks):
        #     print(f"right_hand_cam_landmarks[{index}].x : {cam_landmarks.x}")
    else:
        right_hand_cam_landmarks = None
    
    return pose_cam_landmarks, left_hand_cam_landmarks, right_hand_cam_landmarks

# Holisticモジュールを使って画像を処理
with mp_holistic.Holistic(static_image_mode=True) as holistic:
    results = holistic.process(image_rgb)
    results2 = holistic.process(image2_rgb)

    # 結果の可視化
    if results.pose_landmarks: # 画像１のレンダリング
        print("pose through")
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0))) 
    if results.left_hand_landmarks:
        print("left hand through")
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))
    if results.right_hand_landmarks:
        print("right hand through")
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))
    
    if results2.pose_landmarks: # 画像２のレンダリング
        print("pose 2 through")
        mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))  
    if results2.left_hand_landmarks: 
        print("left hand 2 through")
        mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))
    if results2.right_hand_landmarks:
        print("right hand 2 through")
        mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))  
    
    # 出力したキーポイントをカメラ座標系に変換(x,yの値のみ)
    pose_cam, left_hand_cam, right_hand_cam = transform_result(results, image) 
    pose_cam2, left_hand_cam2, right_hand_cam2 = transform_result(results2, image2)
    stereo_vision_easy(pose_cam, pose_cam2, right_hand_cam, right_hand_cam2, left_hand_cam, left_hand_cam2)     
        

# 処理結果を表示
cv2.imshow('Holistic Result', image)
cv2.imshow('Holistic Result 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()