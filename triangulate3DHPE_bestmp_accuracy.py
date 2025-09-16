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
MediaPipe Poseの高い信頼度をもつランドマーク精度を検証する
入力として"best_landmarks.json"と"hdPose3d_stage1_coco19"の各フレームの結果を用いる
"""
# MediaPipe Holisticモジュールを初期化
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

# パスまとめ
TEMPORARY_IMAGES_STORAGE_PATH = "./output_images/images_temporary_storage/" # 一時的に画像を保存するフォルダのパス
INPUT1_IMAGE_PATH = "./inputs/input_images/"+DATASET_NAME+"_cam"+str(VIDEO1_NUM).zfill(2)+"/"
INPUT2_IMAGE_PATH = "./inputs/input_images/"+DATASET_NAME+"_cam"+str(VIDEO2_NUM).zfill(2)+"/"
VIDEO_STORAGE_PATH = ".outputs/output_videos/"+DATASET_NAME+"_hdvideo"+str(VIDEO1_NUM).zfill(2)+str(VIDEO2_NUM).zfill(2)+"/"
OUTPUT_LANDMARKS_PATH = "outputs/output_landmarks/"+DATASET_NAME+"_cam"+str(VIDEO1_NUM).zfill(2)+str(VIDEO2_NUM).zfill(2)+"/"

# 設定パラメータ
START_FRAME = 137
END_FRAME = 9056
FRAME_INTERVAL = 100
NUM_VIEWS = 31  # 0〜30
NUM_LANDMARKS = 33
EXCLUDED_VIEWS = [20, 21]

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

def project_point(X, R, t, K):
    """
    3D点Xをビューに投影する

    Parameters:
    - X: (3,) ndarray, ワールド座標の3D点
    - R: (3,3) ndarray, 回転行列
    - t: (3,) ndarray, 並進ベクトル
    - K: (3,3) ndarray, カメラ内部パラメータ行列

    Returns:
    - x: (2,) ndarray, 画像座標
    """
    # 3D点をカメラ座標系に変換
    X_cam = R @ X + t  # shape: (3,)

    # 投影（内部パラメータを適用）
    x_proj = K @ X_cam  # shape: (3,)

    # 同次座標を正規化
    x_img = x_proj[:2] / x_proj[2]

    return x_img

"""
-------------------------------------------------------------------------------------------------------
"""
# メイン処理部
if __name__ == "__main__":
    # キャリブレーションファイル読み込み
    with open(f"./panoptic-toolbox/{DATASET_NAME}/calibration_{DATASET_NAME}.json") as calibration_file:
        parameters = json.load(calibration_file)
    print("cam_params:", not(parameters is None))

    # カメラパラメータ取得（HDカメラのみ）
    K_all = [params["K"] for params in parameters["cameras"] if params["type"] == "hd"]
    R_all = [params["R"] for params in parameters["cameras"] if params["type"] == "hd"]
    t_all = [params["t"] for params in parameters["cameras"] if params["type"] == "hd"]

    # CMU PanopticとMediaPipe Pose との対応ランドマークID
    GT_MP_MAP = [
        [1, 0],    # Nose
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

    for frame_num in range(FIRST_FRAME_NUM, END_FRAME_NUM+1, FRAME_INTERVAL):
        # ベストランドマーク呼び出し
        with open("./outputs/best_landmarks/best_landmarks.json", "r") as f:
            data = json.load(f)
        frame_data = data.get(str(frame_num))
        lm_0 = frame_data["0"]

        # GTランドマーク呼び出し
        with open(f"{GT_DATA_FOLDER_PATH}body3DScene_{str(frame_num).zfill(8)}.json", "r") as f:
            data = json.load(f)
        bodies = data.get("bodies")
        landmarks_3d = np.array(bodies[0]["joints19"] if len(bodies) is not 0 else []).reshape(-1, 4)

        # イレギュラーが存在しないフレームのみ計測
        mpjpe_flag = True
        if landmarks_3d.shape[0] == 0:
            mpjpe_flag = False
        for lm in landmarks_3d:
            if lm[3] <= 0.1:
                mpjpe_flag = False
                break
                
        # 誤差格納用
        errors = {
            "holistic": []
        }

        # 使用するGTランドマークリスト
        GT_LANDMARK = landmarks_3d

        # MPJPE 計算処理
        print(f"frame_num:{frame_num}")
        if mpjpe_flag:
            for gt_id, mp_id in GT_MP_MAP:
                lm_info = frame_data[str(mp_id)]
                view_id = lm_info["view"]
                gt_2d = np.array([lm_info["x"], lm_info["y"]])
                
                X_world = GT_LANDMARK[gt_id,:3]
                R = np.array(R_all[view_id])
                t = np.array(t_all[view_id]).reshape(3,)
                K = np.array(K_all[view_id])

                projected_2d = project_point(X_world, R, t, K)
                error = np.linalg.norm(projected_2d - gt_2d)

                # 対応する部位に分類
                # print("error added!")
                errors["holistic"].append(error)

            # === MPJPE 出力 ===
            for part, part_errors in errors.items():
                if part_errors:
                    mpjpe = np.mean(part_errors)
                    print(f"{part} MPJPE: {mpjpe:.2f} cm")
                else:
                    print(f"{part}: ランドマークが見つかりませんでした。")
        else:
            print("GTランドマークが見つかりませんでした")
