import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 
import sys
import json
import glob
import os
from typing import List

"""
結果比較用プログラム
入力画像2枚、3Dプロットを表示
"""
    # if test_mode:
    #     test_root = "./testIO"
    #     inputs_root = os.path.join(test_root, "test_sys_inputs")
    #     videos_root = os.path.join(test_root, "test_sys_outputs", "output_videos")
    #     landmarks_root = os.path.join(test_root, "test_sys_outputs", "output_landmarks")
    #     temp_storage_path = os.path.join(test_root, "test_sys_outputs", "images_temporary_storage")
        
    #     # Create test directories
    #     os.makedirs(inputs_root, exist_ok=True)
    #     os.makedirs(videos_root, exist_ok=True)
    #     os.makedirs(landmarks_root, exist_ok=True)
    #     os.makedirs(temp_storage_path, exist_ok=True)

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# デフォルト設定値
DEFAULT_CAPTURE_RATE = 100
# データセットに含まれるランドマークデータが137フレーム目以降からファイルが存在するため
DEFAULT_FIRST_FRAME_NUM = 137
DEFAULT_DATASET_NAME = "171204_pose3"
DEFAULT_VIDEO1_NUM = 18
DEFAULT_VIDEO2_NUM = 23
DEFAULT_END_FRAME_NUM = 9056
# 一時的に画像を保存するフォルダのパス
DEFAULT_TEMPORARY_IMAGES_STORAGE_PATH = "./output_images/images_temporary_storage/"
DEFAULT_PANOPTIC_ROOT = "./panoptic-toolbox"
DEFAULT_INPUTS_ROOT = "./sys_inputs/input_images"
DEFAULT_VIDEOS_ROOT = "./sys_outputs/output_videos"
DEFAULT_LANDMARKS_ROOT = "./sys_outputs/output_landmarks"
TEMPORARY_IMAGES_STORAGE_PATH = DEFAULT_TEMPORARY_IMAGES_STORAGE_PATH

# 別データ間で同じ箇所を示す対応ランドマーク番号リスト[正解データ, 推定データ]
BODY_MAP = [[1, 0],    # Nose
    [3, 11],   # Left Shoulder
    [4, 13],   # Left Elbow
    [5, 15],   # Left Wrist
    [6, 23],   # Left Hip
    [7, 25],   # Left Knee
    [8, 27],   # Left Ankle
    [9, 12],   # Right Shoulder
    [10, 14],  # Right Elbow
    [11, 16],  # Right Wrist
    [12, 24],  # Right Hip
    [13, 26],  # Right Knee
    [14, 28],  # Right Ankle
    [15, 2],   # Left Eye
    [16, 7],   # Left Ear
    [17, 5],   # Right Eye
    [18, 8],   # Right Ear
    ]

RIGHT_HAND_MAP = [[0, 16],  # Wrist
            [17, 18],   # pinky
            [5, 20],    # index
            [2, 22]]    # thumb

LEFT_HAND_MAP = [[0,15],    # Wrist
                 [17, 17],  # pinky
                 [5, 19],   # index
                 [2, 21]]   # thumb

"""
GT (CMU Panoptic Dataset)
0: Neck
1: Nose
2: BodyCenter (center of hips)
3: lShoulder
4: lElbow
5: lWrist,
6: lHip
7: lKnee
8: lAnkle
9: rShoulder
10: rElbow
11: rWrist
12: rHip
13: rKnee
14: rAnkle
15: lEye
16: lEar
17: rEye
18: rEar

mp (MediaPipe Pose)
0 - nose                       17 - left pinky
1 - left eye (inner)           18 - right pinky
2 - left eye                   19 - left index
3 - left eye (outer)           20 - right index
4 - right eye (inner)          21 - left thumb
5 - right eye                  22 - right thumb
6 - right eye (outer)          23 - left hip
7 - left ear                   24 - right hip
8 - right ear                  25 - left knee
9 - mouth (left)               26 - right knee
10 - mouth (right)             27 - left ankle
11 - left shoulder             28 - right ankle
12 - right shoulder            29 - left heel
13 - left elbow                30 - right heel
14 - right elbow               31 - left foot index
15 - left wrist                32 - right foot index
16 - right wrist

"""

# キャプチャーするフレーム
CAPTURE_RATE = 100  
FIRST_FRAME_NUM = 837 # データセットに含まれるランドマークデータが137フレーム目以降からファイルが存在するため
"""
使用できないフレームについて
下記のフレームは例外となるデータであるため、次のフレームをセットして再度プログラムを実行する．
初期値 137
8337: 手のデータが右手しかなく、ほとんどのランドマークがエラーを示している
"""

# 使用データ指定
DATASET_NAME = "171204_pose3"
GT_DATA_FOLDER_PATH = "./panoptic-toolbox/"+DATASET_NAME+"/hdPose3d_stage1_coco19/"
GT_HAND_DATA_FOLDER_PATH = "./panoptic-toolbox/"+DATASET_NAME+"/hdHand3d/"
VIDEO1_NUM = 18
VIDEO2_NUM = 23
END_FRAME_NUM = 9056

# キャリブレーションファイルオープン
with open("./panoptic-toolbox/"+DATASET_NAME+"/calibration_"+DATASET_NAME+".json") as calibration_file:
    parameters = json.load(calibration_file)
print("cam_params:", not(parameters is None))
# 内部パラメータ、外部パラメータ
param1 = parameters["cameras"][479+VIDEO1_NUM] # HDカメラ00_00 [479]
param2 = parameters["cameras"][479+VIDEO2_NUM]
K_left = np.array(param1["K"])
R_left, T_left = np.array(param1["R"]), np.array(param1["t"])
K_right = np.array(param2["K"])
R_right, T_right = np.array(param2["R"]), np.array(param2["t"])

# パスまとめ
TEMPORARY_IMAGES_STORAGE_PATH = "./output_images/images_temporary_storage/" # 一時的に画像を保存するフォルダのパス
INPUT1_IMAGE_PATH = "./inputs/input_images/"+DATASET_NAME+"_cam"+str(VIDEO1_NUM).zfill(2)+"/"
INPUT2_IMAGE_PATH = "./inputs/input_images/"+DATASET_NAME+"_cam"+str(VIDEO2_NUM).zfill(2)+"/"
VIDEO_STORAGE_PATH = ".outputs/output_videos/"+DATASET_NAME+"_hdvideo"+str(VIDEO1_NUM).zfill(2)+str(VIDEO2_NUM).zfill(2)+"/"
OUTPUT_LANDMARKS_PATH = "sys_outputs/output_landmarks/"+DATASET_NAME+"_cam"+str(VIDEO1_NUM).zfill(2)+str(VIDEO2_NUM).zfill(2)+"/"

# ボーン情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
GT_POSE_CONNECTIONS = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1
GT_HAND_CONNECTIONS = np.array([[1,2],[2,3],[3,4],[4,5],
                                [1,6],[6,7],[7,8],[8,9],
                                [1,10],[10,11],[11,12],[12,13],
                                [1,14],[14,15],[15,16],[16,17],
                                [1,18],[18,19],[19,20],[20,21]])-1

# レンダリング情報
JOINT_STYLE = mp_drawing.DrawingSpec(color=(255,0,0), thickness=5, circle_radius=3)
BONE_STYLE = mp_drawing.DrawingSpec(color=(200,200,0), thickness=5)
mp_drawing._VISIBILITY_THRESHOLD = 0.0 # 画像に表示する際のランドマークの信用度閾値 

# 処理関数
def Landmark_detect(image_left, image_right):
    """
    左右画像それぞれの対象の手、顔、全身の3Dポーズを推定する

    Parameters:
    image_left (MatLike)        : 左カメラの画像
    image_right (MatLike)       : 右カメラの画像

    if test_mode:
        test_root = ./testIO
        inputs_root = os.path.join(test_root, test_sys_inputs)
        videos_root = os.path.join(test_root, test_sys_outputs, output_videos)
        landmarks_root = os.path.join(test_root, test_sys_outputs, output_landmarks)
        temp_storage_path = os.path.join(test_root, test_sys_outputs, images_temporary_storage)
        
        # Create test directories
        os.makedirs(inputs_root, exist_ok=True)
        os.makedirs(videos_root, exist_ok=True)
        os.makedirs(landmarks_root, exist_ok=True)
        os.makedirs(temp_storage_path, exist_ok=True)

    Returns:
    result_left (MediaPipe独自クラス)   : 左画像の3Dポーズ
    result_right (MediaPipe独自クラス)  : 右画像の3Dポーズ
    """
    # RGBスケールに変換
    image_leftRGB = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_rightRGB = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

    # ランドマーク推定 静止画入力 -> static_image_mode=True
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        result_left = holistic.process(image_leftRGB)
        result_right = holistic.process(image_rightRGB)

    return result_left, result_right

def Normalized_to_screen_coord(result_L, result_R, width, height):
    """
    左右画像で推定したランドマークを画像座標系（スクリーン座標系）の値に変換する

    Parameters:
    result (NamedTuple)   : 推定したランドマークのリスト。.landmark.x各ランドマークのx座標の値を取り出せる
    width (int)           : 画像の幅
    height (int)          : 画像の高さ

    Returns:
    scr_landmarks (numpy.ndarray)   : n行2列の画像座標系ランドマーク (全身なのでn=33)
    """
    scr_list_L = [] # カメラ座標を記録する配列を生成
    scr_list_R = []
    # 左側画像のランドマークを変換
    for index, landmark in enumerate(result_L.landmark):
        # キーポイントの値をカメラ座標系（画像中央が原点）に変換
        scr_x = landmark.x * width
        scr_y = landmark.y * height
        # リストに挿入
        scr_list_L.append([scr_x, scr_y]) 

    # 右側画像のランドマークを変換
    for index, landmark in enumerate(result_R.landmark):
        # キーポイントの値をカメラ座標系（画像中央が原点）に変換
        scr_x = landmark.x * width
        scr_y = landmark.y * height
        # リストに挿入
        scr_list_R.append([scr_x, scr_y]) 
    
    scr_landmarks_L = np.array(scr_list_L)
    scr_landmarks_R = np.array(scr_list_R)

    return scr_landmarks_L, scr_landmarks_R

def Matching_landmarks(landmark_left, landmark_right, enable):
    """
    左右の対応するランドマークの信頼度から、信頼度の低いランドマークを含むペアを弾くためのリストを生成する処理

    Parameter:
    landmark_left (NamedTuple)  : 推定した左画像のランドマーク。.landmark.visiblity で推定したランドマークに対する信頼度を取り出せる
    landmark_right (NamedTuple) : 推定した右画像のランドマーク
    """
    # 閾値
    visibility_threshold = 0.80  # 80%以上
    matching_list = []
    if enable:
        for l, r in zip(landmark_left.landmark, landmark_right.landmark):
            print("left visiblity:",l.visibility, " ", "right visiblity:", r.visibility)
            if l.visibility > visibility_threshold and r.visibility > visibility_threshold:
                matching_list.append(1)
            else:
                matching_list.append(0)
    else:
        for i in range(len(landmark_left.landmark)):
            matching_list.append(1)
    # print("match_list:", matching_list)

    return matching_list

def Projection_mat_calc(K_left, R_left, T_left, K_right, R_right, T_right):
    """
    左右カメラの投影行列を生成する

    Parameters:
    K (numpy.ndarray)       : 3行3列のカメラパラメータ
    R (numpy.ndarray)       : 3行3列の回転行列
    t (numpy.ndarray)       : 3行1列の並進ベクトル

    Returns:
    Pleft (numpy.ndarray)   : 左画像のカメラ投影行列 (3*4)
    Pright (numpy.ndarray)  : 右画像のカメラ投影行列 (3*4)
    """
    
    
    # 左カメラをワールド座標とするため、左投影行列 P_left = K[I|0] = [K|0]　となる
    Pleft = np.hstack((K_left @ R_left, K_left @ T_left))
    Pright = np.hstack((K_right @ R_right, K_right @ T_right))

    return Pleft, Pright

def Triangulate_3Dpoint(Pleft, Pright, landmark_left, landmark_right):
    """
    カメラの投影行列を使用して特徴点の3D位置を再構築する

    Parameters:
    Pleft (numpy.ndarray)           : 左画像のカメラ投影行列 (3*4)
    Pright (numpy.ndarray)          : 右画像のカメラ投影行列 (3*4)
    landmark_left (numpy.ndarray)   : n行2列の左画像のランドマーク（画像座標系）
    landmark_right (numpy.ndarray)  : n行2列右画像のランドマーク（画像座標系）(全身なのでn=33)
    matches (list of int)           : 画像間で対応するランドマークの有無を示すリスト

    Returns:
    points3D.T (numpy.ndarray)        : 再構築された33行3列の3Dポイントの配列の転置行列
    """
    # マッチングされた特徴点を取得
    # left_list = []
    # right_list = []
    # for idx, match in enumerate(matchlist):
    #     if match == 1:
    #         left_list.append(landmark_left[idx])
    #         right_list.append(landmark_right[idx])
    # left_pts = np.array(left_list).T
    # right_pts = np.array(right_list).T
    
    # 特徴点の3D位置を再構築　（4行n列）
    points4D = cv2.triangulatePoints(Pleft, Pright, landmark_left.T, landmark_right.T)
    
    # 同次座標を3D座標に変換(3行n列)
    points3D = points4D[:3] / points4D[3] # 3行n列

    # landmark3D = np.zeros((len(matchlist), 3))
    # i = 0 # イテレーター
    # for idx, match in enumerate(matchlist):
    #     if match == 1:
    #         landmark3D[idx] = points3D.T[i]
    #         i+=1

    return points3D.T 

def Triangulate3DHPE(img1, img2, K_l, R_l, T_l, K_r, R_r, T_r):
    # 高さ、幅（同じカメラを用いるため片方の画像から取得）
    HEIGHT, WIDTH, _ = img1.shape

    # ランドマーク、マッチングリスト
    result_left, result_right = Landmark_detect(image_left=img1, image_right=img2)

    if result_left.pose_landmarks == None or result_right.pose_landmarks == None:
        print("Enter no human!")
        landmark3D = []
    else:
        # 画像座標系への変換(同時に体のランドマークのみ使用)
        pose_left, pose_right = Normalized_to_screen_coord(result_left.pose_landmarks, result_right.pose_landmarks, WIDTH, HEIGHT)
        print("left=","\n",pose_left)
        print("right=","\n",pose_right)

        # 投影行列を計算
        Pleft, Pright = Projection_mat_calc(K_l, R_l, T_l, K_r, R_r, T_r)

        # 3次元位置を復元
        landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right)
        print("landmark3D:","\n",landmark3D)

    return result_left, result_right, landmark3D
# 結果表示用関数
def annotate_image(image_L, landmarks_L, image_R, landmarks_R, connections, jointstyle, bonestyle):
    """
    mediapipeが推定したランドマークを画像に描画する関数

    Parameters:
    image (MatLike)                         : 描画する画像
    landmarks (NamedTuple)                  : 推定したランドマークのリスト。
    connections (List of List of int)       : ランドマークを繋ぐボーンの連結情報。連結する2つのランドマークの数字がリスト化されている
    jointstyle (mp_drawing.DrawingSpec)     : ランドマークを示す点の描写情報
    bonestyle (mp_drawing.DrawingSpec)      : ランドマークを繋ぐ線の描写情報

    Return:
    original_images (list of MatLike)  : 注釈前の元の画像 (0:左カメラ画像, 1:右カメラ画像)
    annotated_images (list of MatLike) : 注釈後の画像
    """
    cv2.imwrite(TEMPORARY_IMAGES_STORAGE_PATH+"original_L.png", image_L)
    cv2.imwrite(TEMPORARY_IMAGES_STORAGE_PATH+"original_R.png", image_R)
    # cv2.imshow("L original_image", original_images[0])
    mp_drawing.draw_landmarks(image_L, landmarks_L, connections, jointstyle, bonestyle)
    mp_drawing.draw_landmarks(image_R, landmarks_R, connections, jointstyle, bonestyle)  
    cv2.imwrite(TEMPORARY_IMAGES_STORAGE_PATH+"annotated_L.png", image_L)
    cv2.imwrite(TEMPORARY_IMAGES_STORAGE_PATH+"annotated_R.png", image_R)
    # cv2.imshow("L annotated_image", annotated_images[0])
    image_L_annotation = cv2.imread(TEMPORARY_IMAGES_STORAGE_PATH+"annotated_L.png")
    image_R_annotation = cv2.imread(TEMPORARY_IMAGES_STORAGE_PATH+"annotated_R.png")
    return image_L_annotation, image_R_annotation

def plot_2Dskeleton(landmarks, connection):
    x_coords = []
    y_coords = []
    fig = plt.figure()
    ax = fig.add_subplot()
    # ランドマークをプロット
    for i, landmark in enumerate(landmarks):
        x_coords.append(landmark[0])
        y_coords.append(landmark[1])

        fig.text(landmark[0], landmark[1], str(i), color="black", fontsize=8, ha='right', va='bottom')
        
    # キーポイントを描画
    ax.scatter(x_coords, y_coords, color='red')

    # ボーンを描画
    for start_idx, end_idx in connection:
        ax.plot([x_coords[start_idx], x_coords[end_idx]],
                    [y_coords[start_idx], y_coords[end_idx]],
                    color='blue')
        
    ax.invert_yaxis()  # Y座標を反転して画像と一致させる
    ax.axis('equal')         # アスペクト比を正確にする
    # fig.savefig()
    plt.show()
    cv2.destroyAllWindows()

def plot_3Dskeleton(frame_num, output_landmarks, gt_landmarks, gt_left, gt_right, output_connections, gt_connections, gt_hand_connections):
    fig = plt.figure(figsize = (8, 8))
    ax= fig.add_subplot(111, projection='3d')
    
    # 推定結果をプロットに表示
    ax.scatter(output_landmarks[:, 0], output_landmarks[:,1],output_landmarks[:,2], s = 10, c = "red")
    for connection in output_connections:
        start_joint, end_joint = connection
        x = [output_landmarks[start_joint, 0], output_landmarks[end_joint, 0]]
        y = [output_landmarks[start_joint, 1], output_landmarks[end_joint, 1]]
        z = [output_landmarks[start_joint, 2], output_landmarks[end_joint, 2]]
        ax.plot(x, y, z, c='red', linewidth=2)

    # グラウンドトゥルースをプロットに表示
    ax.scatter(gt_landmarks[:, 0], gt_landmarks[:,1],gt_landmarks[:,2], s = 10, c = "blue") # ボディ
    for connection in gt_connections:
        start_joint, end_joint = connection
        x = [gt_landmarks[start_joint, 0], gt_landmarks[end_joint, 0]]
        y = [gt_landmarks[start_joint, 1], gt_landmarks[end_joint, 1]]
        z = [gt_landmarks[start_joint, 2], gt_landmarks[end_joint, 2]]
        ax.plot(x, y, z, c='blue', linewidth=2)
    ax.scatter(gt_left[:, 0], gt_left[:,1],gt_left[:,2], s = 10, c = "blue") # 左手
    for connection in gt_hand_connections:
        start_joint, end_joint = connection
        x = [gt_left[start_joint, 0], gt_left[end_joint, 0]]
        y = [gt_left[start_joint, 1], gt_left[end_joint, 1]]
        z = [gt_left[start_joint, 2], gt_left[end_joint, 2]]
        ax.plot(x, y, z, c='blue', linewidth=2)
    ax.scatter(gt_right[:, 0], gt_right[:,1],gt_right[:,2], s = 10, c = "blue") # 右手
    for connection in gt_hand_connections:
        start_joint, end_joint = connection
        x = [gt_right[start_joint, 0], gt_right[end_joint, 0]]
        y = [gt_right[start_joint, 1], gt_right[end_joint, 1]]
        z = [gt_right[start_joint, 2], gt_right[end_joint, 2]]
        ax.plot(x, y, z, c='blue', linewidth=2)

        # if not (gt_right[start_joint] == [0,0,0] or gt_right[end_joint] == [0,0,0]): # [0,0,0]をプロットから省く
    set_equal_aspect(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("frame"+str(frame_num).zfill(8))
    # 対象を正面から見た3Dプロットを画像として保存
    ax.view_init(elev=180, azim=5, roll=-90)

    return fig, ax

# プロットの軸を等間隔にする
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

     # ボックスアスペクト比を均等に設定 (バージョン 3.4+)
    ax.set_box_aspect([1, 1, 1])  # x, y, z を同じスケールに

def concatenate_images(frame_num):
    # 入力フレームの読み込み
    image1 = cv2.imread(INPUT1_IMAGE_PATH+"frame"+str(frame_num).zfill(8)+".png")
    image2 = cv2.imread(INPUT2_IMAGE_PATH+"frame"+str(frame_num).zfill(8)+".png")

    # 画像を横に連結
    concatenation_image = cv2.hconcat([image2, image1])

    return concatenation_image

def create_video_from_images(concatenation_left, concatenation_right, frame_rate, save_path):
    # 左カメラ各フレームの比較用画像読み込み
    frames_left = []
    for img in concatenation_left:
        if img is None:
            break
        frames_left.append(img)
    
    # 右カメラ各フレームの比較用画像読み込み
    frames_right = []
    for img in concatenation_right:
        if img is None:
            break
        frames_right.append(img)

    HEIGHT, WIDTH, _ = frames_left[0].shape
    out_Lvideo = cv2.VideoWriter(save_path+"output_left_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (WIDTH, HEIGHT))
    out_Rvideo = cv2.VideoWriter(save_path+"output_right_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (WIDTH, HEIGHT)) 

    for i in range(len(frames_left)):
        out_Lvideo.write(frames_left[i])
        out_Rvideo.write(frames_right[i])
# create_video_from_images(storage_path=FRAMES_STORAGE_PATH)

# 集計関数
def compute_mpjpe(predicted, ground_truth):
    """
    MPJPE (Mean Per Joint Position Error) を計算する関数。

    Parameters:
        predicted (np.ndarray): 予測された3D関節位置 (N, 3)
        ground_truth (np.ndarray): GTの3D関節位置 (N, 3)

    Returns:
        float: MPJPE値（mm単位を想定）
    """
    assert predicted.shape == ground_truth.shape, "形状が一致しません"
    errors = np.linalg.norm(predicted - ground_truth, axis=1)
    mpjpe = np.mean(errors)
    return mpjpe

def make_compe_landmarks(ground_truth, predicted, compe_map, confidence:float = -1.0):
    """
    同じ箇所を推定したランドマークを抽出し、各ランドマークを出力する操作
    GTランドマークの信頼度が閾値を下回っているならそのペアを除く

    Parameters:
        predicted (np.ndarray): 予測された3D関節位置 (N, 3)
        ground_truth (np.ndarray): GTの3D関節位置 (N, 3)
        compe_map (np.ndarray): 対応ランドマークのペアマップ (M, 2)
        confidence (float): 信頼度の閾値

    Returns:
        compe_predicted (np.ndarray): 抽出した予測3D関節位置 (N, 3)
        compe_ground_truth (np.ndarray): 抽出したGTの3D関節位置 (N, 3)
    """

    compe_predicted = []
    compe_ground_truth = []


    for [gt_num, pd_num] in compe_map:
        # === GT信頼度フィルタ ===
        if ground_truth [gt_num, 3] <= confidence:
            # print("Throught!!")
            continue

        compe_predicted.append(predicted[pd_num])
        compe_ground_truth.append(ground_truth[gt_num])

    return np.array(compe_ground_truth), np.array(compe_predicted)

def read_gt_landmarks(frame_num):
    # 正解データ呼び出し
    with open(GT_DATA_FOLDER_PATH+"body3DScene_"+str(frame_num).zfill(8)+".json") as gt: # ボディ
        ground_truth_file = gt
        gt_frame = json.load(ground_truth_file)
        print(not(gt_frame is None))
    # (x,y,z,c)の19行4列の配列を作成
    if len(gt_frame['bodies']) == 0:
        gt_body = np.zeros((19, 4))
    else:
        gt_body = np.array(gt_frame['bodies'][0]['joints19']).reshape(-1, 4)
    with open(GT_HAND_DATA_FOLDER_PATH+"handRecon3D_hd"+str(frame_num).zfill(8)+".json") as gt: # 両手
        ground_truth_file = gt
        gt_frame = json.load(ground_truth_file)
        print(not(gt_frame is None))
    # (x,y,z)の21行3列の配列を作成 'landmarks'の長さが63であり、手のランドマークは21個であることから想定
    if len(gt_frame['people']) == 0:
        gt_left = np.zeros((21, 3))
        gt_right = np.zeros((21, 3))
    else:
        if not(len(gt_frame['people'][0]['left_hand']) == 0):
            gt_left = np.array(gt_frame['people'][0]['left_hand']['landmarks']).reshape(-1, 3)
        else:
            gt_left = np.zeros((21,3))
        if not(len(gt_frame['people'][0]['right_hand']) == 0):
            gt_right = np.array(gt_frame['people'][0]['right_hand']['landmarks']).reshape(-1, 3)
        else:
            gt_right = np.zeros((21,3))
    
    return gt_body, gt_left, gt_right

def read_pd_landmarks(frame_num):
    # 推定ランドマーク呼び出し
    with open(OUTPUT_LANDMARKS_PATH+"frame_"+str(frame_num).zfill(8)+".json") as out:
        out_frame_data = json.load(out)
        print(not(out_frame_data is None))
    # (x,y,z)の33行3列の配列を作成
    if out_frame_data['landmarks'] == None:
        out_body = np.zeros((33, 3))
    else:
        out_body_list = []
        for lm_num, lm in out_frame_data['landmarks'].items():
            out_body_list.append(lm)
        out_body = np.array(out_body_list).reshape(-1, 3)
    return out_body

def run_triangulate3dhpe_data(
    dataset_name: str = DEFAULT_DATASET_NAME,
    video1_num: int = DEFAULT_VIDEO1_NUM,
    video2_num: int = DEFAULT_VIDEO2_NUM,
    capture_rate: int = DEFAULT_CAPTURE_RATE,
    first_frame_num: int = DEFAULT_FIRST_FRAME_NUM,
    end_frame_num: int = DEFAULT_END_FRAME_NUM,
    panoptic_root: str = DEFAULT_PANOPTIC_ROOT,
    inputs_root: str = DEFAULT_INPUTS_ROOT,
    videos_root: str = DEFAULT_VIDEOS_ROOT,
    landmarks_root: str = DEFAULT_LANDMARKS_ROOT,
    temp_storage_path: str = DEFAULT_TEMPORARY_IMAGES_STORAGE_PATH,
    test_mode: bool = False,
) -> List[int]:
    """
    指定した設定でMediaPipeを用いた3Dランドマーク推定と保存を実行する。

    Parameters
    ----------
    dataset_name : str
        使用するデータセット名。
    video1_num : int
        左カメラの番号。
    video2_num : int
        右カメラの番号。
    capture_rate : int
        フレームのサンプリング間隔。
    first_frame_num : int
        推定を始めるフレーム番号。
    end_frame_num : int
        推定を終了するフレーム番号（未満まで処理）。
    panoptic_root : str
        Panopticデータセットが保存されているルートパス。
    inputs_root : str
        入力画像を保存するルートパス。
    videos_root : str
        比較用動画を保存するルートパス。
    landmarks_root : str
        ランドマーク推定結果を保存するルートパス。
    temp_storage_path : str
        中間生成物を保存するパス。
    test_mode : bool
        Trueの場合、testIOフォルダ構造を使用。

    Returns
    -------
    List[int]
        推定結果を保存したフレーム番号のリスト。
    """
    # パス生成
    dataset_root = os.path.join(panoptic_root, dataset_name)
    video1_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video1_num).zfill(2)}.mp4")
    video2_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video2_num).zfill(2)}.mp4")
    calibration_path = os.path.join(dataset_root, f"calibration_{dataset_name}.json")

    input1_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}")
    input2_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video2_num).zfill(2)}")
    video_storage_path = os.path.join(videos_root, f"{dataset_name}_hdvideo{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")
    output_landmarks_path = os.path.join(landmarks_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")

    # フォルダ生成
    os.makedirs(video_storage_path, exist_ok=True)
    os.makedirs(input1_image_path, exist_ok=True)
    os.makedirs(input2_image_path, exist_ok=True)
    os.makedirs(output_landmarks_path, exist_ok=True)
    os.makedirs(temp_storage_path, exist_ok=True)

    # キャリブレーションファイルをロード
    with open(calibration_path) as calibration_file:
        parameters = json.load(calibration_file)
    print("cam_params:", parameters is not None)

    param1 = parameters["cameras"][479 + video1_num] # カメラ1
    param2 = parameters["cameras"][479 + video2_num] # カメラ2
    K_left = np.array(param1["K"]) # カメラ1カメラパラメータ
    R_left, T_left = np.array(param1["R"]), np.array(param1["t"]) # カメラ1外部パラメータ
    K_right = np.array(param2["K"]) # カメラ2カメラパラメータ
    R_right, T_right = np.array(param2["R"]), np.array(param2["t"]) # カメラ2外部パラメータ

    video1 = cv2.VideoCapture(video1_path) # カメラ1動画
    video2 = cv2.VideoCapture(video2_path) # カメラ2動画

    if not video1.isOpened() or not video2.isOpened():
        video1.release()
        video2.release()
        raise FileNotFoundError("指定した動画ファイルを開けませんでした。パスを確認してください。")

    processed_frames: List[int] = []
    frame_num = first_frame_num

    try:
        while frame_num < end_frame_num:
            video1.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            video2.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            print("now frame:" + str(video1.get(cv2.CAP_PROP_POS_FRAMES)))

            ret1, img1 = video1.read()
            ret2, img2 = video2.read()
            print("now frame:" + str(video1.get(cv2.CAP_PROP_POS_FRAMES)))

            if not (ret1 and ret2):
                print("breaked frame:", frame_num)
                break

            cv2.imwrite(os.path.join(input1_image_path, f"frame{frame_num:08d}.png"), img1)
            cv2.imwrite(os.path.join(input2_image_path, f"frame{frame_num:08d}.png"), img2)

            _, _, landmarks = Triangulate3DHPE(img1, img2, K_left, R_left, T_left, K_right, R_right, T_right)

            landmark_dict = None
            if len(landmarks) != 0:
                landmark_dict = {str(idx): [float(landmark[0]), float(landmark[1]), float(landmark[2])]
                                 for idx, landmark in enumerate(landmarks)}

            result = {"frame": frame_num, "landmarks": landmark_dict}
            with open(os.path.join(output_landmarks_path, f"frame_{frame_num:08d}.json"), "w") as f:
                json.dump(result, f, indent=2)

            processed_frames.append(frame_num)
            frame_num += capture_rate
    finally:
        video1.release()
        video2.release()

    print("finish!!")
    return processed_frames    

def run_mpjpe_evaluation(
    first_frame: int = DEFAULT_FIRST_FRAME_NUM,
    end_frame: int = DEFAULT_END_FRAME_NUM,
    capture_rate: int = DEFAULT_CAPTURE_RATE,
    gt_body_folder: str = GT_DATA_FOLDER_PATH,
    gt_hand_folder: str = GT_HAND_DATA_FOLDER_PATH,
    prediction_folder: str = OUTPUT_LANDMARKS_PATH,
    body_map: list[list[int]] = BODY_MAP,
    left_hand_map: list[list[int]] = LEFT_HAND_MAP,
    right_hand_map: list[list[int]] = RIGHT_HAND_MAP,
    confidence_threshold: float = 0.0,
) -> list[dict]:
    """
    trianguate3DHPE_compute_mpjpe copy.py のメイン処理（全身・左右手・ホリスティックの MPJPE 計測）を
    関数化したラッパー。指定フレームの範囲で GT と推定値を読み込み、MPJPE をフレームごとに計算して返す。

    Returns
    -------
    list[dict]
        各フレームの計算結果をまとめた辞書のリスト。
        例:
        {
            "frame": 8437,
            "body_mpjpe": 42.18,
            "left_hand_mpjpe": 35.72,
            "right_hand_mpjpe": 37.01,
            "holistic_mpjpe": 39.85,
            "status": "ok"
        }
        読み込み失敗や GT 欠落時は `status` に "missing_gt" / "missing_prediction" などが入る。
    """
    results: list[dict] = []
    frame_num = first_frame

    # # パス生成
    # dataset_root = os.path.join(panoptic_root, dataset_name)
    # video1_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video1_num).zfill(2)}.mp4")
    # video2_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video2_num).zfill(2)}.mp4")
    # calibration_path = os.path.join(dataset_root, f"calibration_{dataset_name}.json")

    # input1_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}")
    # input2_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video2_num).zfill(2)}")
    # video_storage_path = os.path.join(videos_root, f"{dataset_name}_hdvideo{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")
    # output_landmarks_path = os.path.join(landmarks_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")

    while frame_num < end_frame:
        result_row: dict = {"frame": frame_num, "status": "ok"}
        mpjpe_flag = True

        body_path = os.path.join(gt_body_folder, f"body3DScene_{frame_num:08d}.json")
        hand_path = os.path.join(gt_hand_folder, f"handRecon3D_hd{frame_num:08d}.json")
        pred_path = os.path.join(prediction_folder, f"frame_{frame_num:08d}.json")

        try:
            with open(body_path) as body_file:
                gt_body_json = json.load(body_file)
        except FileNotFoundError:
            result_row["status"] = "missing_gt_body"
            results.append(result_row)
            frame_num += capture_rate
            continue

        if len(gt_body_json.get("bodies", [])) == 0:
            gt_body = np.zeros((19, 4))
            mpjpe_flag = False
        else:
            gt_body = np.asarray(gt_body_json["bodies"][0]["joints19"]).reshape(-1, 4)

        try:
            with open(hand_path) as hand_file:
                gt_hand_json = json.load(hand_file)
        except FileNotFoundError:
            result_row["status"] = "missing_gt_hand"
            results.append(result_row)
            frame_num += capture_rate
            continue

        if len(gt_hand_json.get("people", [])) == 0:
            gt_left = np.zeros((21, 4))
            gt_right = np.zeros((21, 4))
            mpjpe_flag = False
        else:
            person = gt_hand_json["people"][0]

            if person.get("left_hand"):
                left_landmarks = np.asarray(person["left_hand"]["landmarks"]).reshape(-1, 3)
                left_conf = np.asarray(person["left_hand"]["averageScore"]).reshape(-1, 1)
                gt_left = np.concatenate((left_landmarks, left_conf), axis=1)
            else:
                gt_left = np.zeros((21, 4))
                mpjpe_flag = False

            if person.get("right_hand"):
                right_landmarks = np.asarray(person["right_hand"]["landmarks"]).reshape(-1, 3)
                right_conf = np.asarray(person["right_hand"]["averageScore"]).reshape(-1, 1)
                gt_right = np.concatenate((right_landmarks, right_conf), axis=1)
            else:
                gt_right = np.zeros((21, 4))
                mpjpe_flag = False

        try:
            with open(pred_path) as pred_file:
                pred_json = json.load(pred_file)
        except FileNotFoundError:
            result_row["status"] = "missing_prediction"
            results.append(result_row)
            frame_num += capture_rate
            continue

        if pred_json.get("landmarks") is None:
            out_body = np.zeros((33, 3))
            mpjpe_flag = False
        else:
            # 保存されている33個のランドマークをインデックス順に並べる
            out_body = np.asarray([pred_json["landmarks"][str(idx)] for idx in range(33)], dtype=float)

        compe_gt, compe_pd = make_compe_landmarks(gt_body, out_body, body_map, confidence=confidence_threshold)
        left_compe_gt, left_compe_pd = make_compe_landmarks(gt_left, out_body, left_hand_map, confidence=confidence_threshold)
        right_compe_gt, right_compe_pd = make_compe_landmarks(gt_right, out_body, right_hand_map, confidence=confidence_threshold)

        if not mpjpe_flag:
            result_row["status"] = "missing_gt"
        else:
            if compe_gt.size > 0 and compe_pd.size > 0:
                result_row["body_mpjpe"] = float(compute_mpjpe(compe_pd, compe_gt[:, :3]))
            if left_compe_gt.size > 0 and left_compe_pd.size > 0:
                result_row["left_hand_mpjpe"] = float(compute_mpjpe(left_compe_pd, left_compe_gt[:, :3]))
            if right_compe_gt.size > 0 and right_compe_pd.size > 0:
                result_row["right_hand_mpjpe"] = float(compute_mpjpe(right_compe_pd, right_compe_gt[:, :3]))

            if (
                compe_gt.size > 0 and compe_pd.size > 0
                and left_compe_gt.size > 0 and left_compe_pd.size > 0
                and right_compe_gt.size > 0 and right_compe_pd.size > 0
            ):
                holistic_gt = np.concatenate((compe_gt, left_compe_gt, right_compe_gt), axis=0)
                holistic_pd = np.concatenate((compe_pd, left_compe_pd, right_compe_pd), axis=0)
                result_row["holistic_mpjpe"] = float(compute_mpjpe(holistic_pd, holistic_gt[:, :3]))
            else:
                result_row.setdefault("holistic_mpjpe", None)

        results.append(result_row)
        frame_num += capture_rate

    return results

if __name__ == "__main__": # テスト用
    results = run_mpjpe_evaluation(capture_rate=30)
    print("finished!!")
    for i in range(0, len(results)):
        if results[i]["status"] == 'ok':   
            print(f"{results[i]["right_hand_mpjpe"]:.2f}")
        else:
            print(results[i]['status'])

