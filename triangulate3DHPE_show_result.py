import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 
import sys
import json
import glob
import os

"""
結果比較用プログラム
入力画像2枚、3Dプロットを表示
"""
# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# キャプチャーするフレーム
CAPTURE_RATE = 100  
FIRST_FRAME_NUM = 137 # データセットに含まれるランドマークデータが137フレーム目以降からファイルが存在するため
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
OUTPUT_LANDMARKS_PATH = "outputs/output_landmarks/"+DATASET_NAME+"_cam"+str(VIDEO1_NUM).zfill(2)+str(VIDEO2_NUM).zfill(2)+"/"

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
    mp_drawing.draw_landmarks(image_L, landmarks_L, connections, 
                            jointstyle, bonestyle)
    mp_drawing.draw_landmarks(image_R, landmarks_R, connections, 
                        jointstyle, bonestyle)  
    cv2.imwrite(TEMPORARY_IMAGES_STORAGE_PATH+"annotated_L.png", image_L)
    cv2.imwrite(TEMPORARY_IMAGES_STORAGE_PATH+"annotated_R.png", image_R)
    # cv2.imshow("L annotated_image", annotated_images[0])

def plot_2Dskeleton(landmarks, connection):
    x_coords = []
    y_coords = []
    # ランドマークをプロット
    for i, landmark in enumerate(landmarks):
        x_coords.append(landmark[0])
        y_coords.append(landmark[1])

        plt.text(landmark[0], landmark[1], str(i), color="black", fontsize=8, ha='right', va='bottom')
        
    # キーポイントを描画
    plt.scatter(x_coords, y_coords, color='red')

    # ボーンを描画
    for start_idx, end_idx in connection:
        plt.plot([x_coords[start_idx], x_coords[end_idx]],
                    [y_coords[start_idx], y_coords[end_idx]],
                    color='blue')
        
    plt.gca().invert_yaxis()  # Y座標を反転して画像と一致させる
    plt.axis('equal')         # アスペクト比を正確にする
    fig.savefig()
    plt.show()

def plot_3Dskeleton(output_landmarks, gt_landmarks, gt_left, gt_right, output_connections, gt_connections, gt_hand_connections):
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

"""
-------------------------------------------------------------------------------------------------------
"""
# メイン処理部
frame_num = FIRST_FRAME_NUM
while frame_num < END_FRAME_NUM:
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
       

    # 推定データ呼び出し
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
    
    # 画像、プロット表示
    concat_image = concatenate_images(frame_num) # 比較しやすいように入力画像を連結
    plot_3Dskeleton(out_body, gt_body, gt_left, gt_right, POSE_CONNECTIONS, GT_POSE_CONNECTIONS, GT_HAND_CONNECTIONS)
    cv2.imshow("frame"+str(frame_num).zfill(8), concat_image) 
    plt.show()
    
    plt.clf()
    plt.close()
    cv2.destroyAllWindows() 

    frame_num += CAPTURE_RATE

cv2.destroyAllWindows()
