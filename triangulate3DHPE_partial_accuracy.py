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
FIRST_FRAME_NUM = 3037 # データセットに含まれるランドマークデータが137フレーム目以降からファイルが存在するため
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

# 別データ間で同じ箇所を示す対応ランドマーク番号リスト[正解データ, 推定データ]
GT_PR_MAP = [[1, 0],    # Nose
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
# 結果表示用関数
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

def make_compe_landmarks(ground_truth, predicted, compe_map):
    """
    同じ箇所を推定したランドマークを抽出し、各ランドマークを出力する操作

    Parameters:
        predicted (np.ndarray): 予測された3D関節位置 (N, 3)
        ground_truth (np.ndarray): GTの3D関節位置 (N, 3)
        compe_map (np.ndarray): 対応ランドマークのペアマップ (M, 2)

    Returns:
        compe_predicted (np.ndarray): 抽出した予測3D関節位置 (N, 3)
        compe_ground_truth (np.ndarray): 抽出したGTの3D関節位置 (N, 3)
    """

    compe_predicted = []
    compe_ground_truth = []

    for [gt_num, pd_num] in compe_map:
        compe_predicted.append(predicted[pd_num])
        compe_ground_truth.append(ground_truth[gt_num])

    return np.array(compe_ground_truth), np.array(compe_predicted)

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
        out_body = np.array(out_frame_data['landmarks']).reshape(-1, 3)
    
    # 画像、プロット表示
    compe_gt, compe_pd = make_compe_landmarks(gt_body, out_body, GT_PR_MAP)
    mpjpe = compute_mpjpe(compe_pd, compe_gt[:,:3]) # 4列目（信頼度）は除く
    print("frame"+str(frame_num).zfill(4)+", mpjpe:"+str(mpjpe))

    frame_num += CAPTURE_RATE

cv2.destroyAllWindows()
