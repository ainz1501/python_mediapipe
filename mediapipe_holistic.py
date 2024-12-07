import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 骨格情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# レンダリング情報
JOINT_STYLE = mp_drawing.DrawingSpec(color=(0,0,255), thickness=30, circle_radius=10)
BONE_STYLE = mp_drawing.DrawingSpec(color=(200,200,0), thickness=15)

# 画像読み込み サイズは4284 × 5712
IMG_PATH = '/Users/tokudataichi/Documents/python_mediapipe/input_images/left0_image.JPG'
image = cv2.imread(IMG_PATH)
IMG2_PATH = '/Users/tokudataichi/Documents/python_mediapipe/input_images/right50_image.JPG'
image2 = cv2.imread(IMG2_PATH)

# 高さ、幅（同じカメラを用いるため片方の画像から取得）
HEIGHT, WIDTH, _ = image.shape

# BGRをRGBに変換（MediaPipeがRGBを期待するため）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# スクリーン座標系キーポイント構造体
class scr_landmark:
    x:float
    y:float

    def __init__(self, input_x, input_y) -> None:
        self.x = input_x
        self.y = input_y

# 平行ステレオビジョン
def stereo_vision_parallel(pose1, pose2, left1, left2, right1, right2):
    stereo_Stime = datetime.now()
    pose_3d_landmarks = []
    right_hand_3d = []
    left_hand_3d = []
    baseline = 500.0 # 単位は[mm]
    focal_length = 26.0 # 単位は[mm]
    pixel_pitch = 0.002 #　単位は[mm]

    # ３Dジョイント位置計算
    if pose1 and pose2: 
        for p1, p2 in zip(pose1, pose2):
            joint_x = (baseline*p1.x)/(p1.x-p2.x) # 単位は全て[mm]
            joint_y = (baseline*p1.y)/(p1.x-p2.x)
            joint_z = (focal_length*baseline)/(pixel_pitch*(p1.x-p2.x))

            print(f"x:{joint_x}, y:{joint_y}, z:{joint_z}")
            pose_3d_landmarks.append([joint_x, joint_y, joint_z])
        pose_3d_landmarks = np.array(pose_3d_landmarks)
    
    stereo_Etime = datetime.now()
    print(f"stereo time:{stereo_Etime - stereo_Stime}")

    return pose_3d_landmarks

# 外部パラメータ演算関数
def estimate_external_params(left, right, K):
    
    # 基本行列
    E, _ = cv2.findEssentialMat(left, right, K, cv2.RANSAC)
    # 回転行列、並進ベクトル
    _, R, T, _ = cv2.recoverPose(E, left, right)

    # SVDによるR,Tが正しいのかを判定
    print(f"R={R[:3]}")
    print(f"T={T[:3]}")
    # print(f"SVD chack:{np.cross(R,T) == E}")

    return R, T 

# ３D位置復元
def triangulate_points(P1, P2, left, right):
    match_list = []

    for idx, (l, r) in enumerate(zip(left, right)):
        if not(l.size == 0 or r.size == 0): # 対応するキーポイントが左右とも存在する場合に限定
            match_list.append([idx, l, r])
    
    points4D = cv2.triangulatePoints(P1, P2, match_list[:, 1], match_list[:, 2])
    points3D =  points4D[:3] / points4D[3]
    pointsID = match_list[:, 0]
    return points3D, pointsID

# ステレオビジョン
def stereo_vision(pose_left, pose_right, width, height):
    stereo_Stime = datetime.now()
    focal_length = 26.0 # 単位は[mm]
    pixel_pitch = 0.002 #　単位は[mm]

    # カメラ内部パラメータ　（仮置き）
    internal_param = np.array([[focal_length/pixel_pitch, 0, width/2],
                                [0, focal_length/pixel_pitch, height/2],
                                [0, 0, 1]])
    
    # 左カメラの外部パラメータと投影行列
    R_left = np.eye(3)
    T_left = np.zeros((3, 1))
    P_left = np.dot(internal_param, np.hstack((R_left, T_left)))

    # 右カメラの外部パラメータと投影行列
    R_right, T_right = estimate_external_params(pose_left, pose_right, internal_param)
    P_right = np.dot(internal_param, np.hstack((R_right, T_right)))

    # 3次元ジョイントの復元
    pose_3d_coordinate, pose_3d_ID = triangulate_points(P_left, P_right, pose_left, pose_right)
    pose_3d_landmarks = np.hstack((pose_3d_ID.T, pose_3d_coordinate))
    stereo_Etime = datetime.now()
    print(f"stereo time:{stereo_Etime - stereo_Stime}")
    return pose_3d_landmarks

# スクリーン座標への変換（同一デバイス使用前提）
def transform2screen(result, width, height):
    scr_list = [] # カメラ座標を記録する配列を生成
    for index, landmark in enumerate(result.landmark):
        # キーポイントの値をカメラ座標系（画像中央が原点）に変換
        scr_x = landmark.x * width
        scr_y = landmark.y * height
        # scr_landmarkのインスタンスの生成と同時に、x座標、y座標の値を入力
        scr_list.append([scr_x, scr_y]) 
        # print(f"scr_landmarks[{index}].x: {scr_landmarks[index].x}")
    
    scr_landmarks = np.array(scr_list)

    return scr_landmarks

        
# キーポイント座標の変換処理まとめ
def transform_result(results, width, height):

    if results.pose_landmarks: # ボディランドマーク変換
        pose_scr_landmarks = transform2screen(results.pose_landmarks, width, height)
        # rectification_pose = stereo_rectification(pose_scr_landmarks)
        # for index, scr_landmarks in enumerate(pose_scr_landmarks):
        #     print(f"pose_scr_landmarks[{index}].x : {scr_landmarks.x}")
    else:
        pose_scr_landmarks = None
        rectification_pose = None


    if results.left_hand_landmarks:
        left_hand_scr_landmarks = transform2screen(results.left_hand_landmarks, width, height)
        # rectification_left_hand = stereo_rectification(pose_scr_landmarks)
        # for index, scr_landmarks in enumerate(left_hand_scr_landmarks):
        #     print(f"left_hand_scr_landmarks[{index}].x : {scr_landmarks.x}")
    else:
        left_hand_scr_landmarks = None
    if results.right_hand_landmarks:
        right_hand_scr_landmarks = transform2screen(results.right_hand_landmarks, width, height)
        # rectification_right_hand = stereo_rectification(pose_scr_landmarks)
        # for index, scr_landmarks in enumerate(right_hand_scr_landmarks):
        #     print(f"right_hand_scr_landmarks[{index}].x : {scr_landmarks.x}")
    else:
        right_hand_scr_landmarks = None
    
    return pose_scr_landmarks, left_hand_scr_landmarks, right_hand_scr_landmarks



# メイン処理
with mp_holistic.Holistic(static_image_mode=True) as holistic:
    all_process_Stime = datetime.now()
    mediapipe_Stime = datetime.now()
    results = holistic.process(image_rgb)
    results2 = holistic.process(image2_rgb)
    mediapipe_Etime = datetime.now()
    print(f"mediapipe time:{mediapipe_Etime - mediapipe_Stime}")
    
    # 出力したキーポイントをカメラ座標系に変換(x,yの値のみ)
    pose_scr, left_hand_scr, right_hand_scr = transform_result(results, WIDTH, HEIGHT) 
    pose_scr2, left_hand_scr2, right_hand_scr2 = transform_result(results2, WIDTH, HEIGHT)

    # 3Dキーポイント復元
    pose_3d_landmarks = stereo_vision(pose_scr, pose_scr2, WIDTH, HEIGHT) 

    all_process_Etime = datetime.now()
    print(f"all process time:{all_process_Etime - all_process_Stime}")


# レンダリング処理
rendering_Stime = datetime.now()
# ２Dスケルトンレンダリング処理
if results.pose_landmarks: # 画像１のレンダリング
    print("pose through")
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                JOINT_STYLE, BONE_STYLE) 
# if results.left_hand_landmarks:
#     print("left hand through")
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)
# if results.right_hand_landmarks:
#     print("right hand through")
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)

if results2.pose_landmarks: # 画像２のレンダリング
    print("pose 2 through")
    mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                JOINT_STYLE, BONE_STYLE)  
# if results2.left_hand_landmarks: 
#     print("left hand 2 through")
#     mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)
# if results2.right_hand_landmarks:
#     print("right hand 2 through")
#     mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)  
cv2.imshow('Holistic Result', image)
cv2.imshow('Holistic Result 2', image2)

# 3Dスケルトンプロット生成処理
fig = plt.figure(figsize = (8, 8))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(pose_3d_landmarks[:, 0], pose_3d_landmarks[:,1],pose_3d_landmarks[:,2], s = 1, c = "blue")
# 骨格情報からボーンを形成
for connection in POSE_CONNECTIONS:
    start_joint, end_joint = connection
    x = [pose_3d_landmarks[start_joint, 0], pose_3d_landmarks[end_joint, 0]]
    y = [pose_3d_landmarks[start_joint, 1], pose_3d_landmarks[end_joint, 1]]
    z = [pose_3d_landmarks[start_joint, 2], pose_3d_landmarks[end_joint, 2]]
    plt.plot(x, y, z, c='red', linewidth=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.autoscale(None)
plt.title("3D Skeleton Visualization")
rendering_Etime = datetime.now()
print(f"rendering time:{rendering_Etime - rendering_Stime}")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# メイン終了