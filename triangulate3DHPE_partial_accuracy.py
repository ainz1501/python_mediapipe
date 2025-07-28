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

    for frame_num in range(FIRST_FRAME_NUM, END_FRAME_NUM+1, FRAME_INTERVAL):
        # ベストランドマーク呼び出し
        with open("./outputs/best_landmarks/best_landmarks.json", "r") as f:
            data = json.load(f)
        frame_data = data.get(str(frame_num))

        # 推定ランドマーク
        with open(f"./{OUTPUT_LANDMARKS_PATH}frame_{str(frame_num).zfill(8)}.json", "r") as f:
            data = json.load(f)
        landmarks_3d = data.get("landmarks")

        # MediaPipeの部位ごとのランドマークID（文字列で）
        right_hand_ids = [str(i) for i in [15, 17, 19, 21]]  # 右手：右手首, 右親指先, 右人差し指先, 右小指先
        left_hand_ids  = [str(i) for i in [16, 18, 20, 22]]  # 左手：左手首, 左親指先, 左人差し指先, 左小指先
        right_leg_ids  = [str(i) for i in [24, 28, 32]]  # 右足：腰, 膝, 足首, 足先
        left_leg_ids   = [str(i) for i in [23, 27, 31]]  # 左足：腰, 膝, 足首, 足先

        # 誤差格納用
        errors = {
            "right_hand": [],
            "left_hand": [],
            "right_leg": [],
            "left_leg": []
        }

        for lm_id, lm_info in frame_data.items():
            view_id = lm_info["view"]
            gt_2d = np.array([lm_info["x"], lm_info["y"]])

            if lm_id not in landmarks_3d:
                continue
            
            X_world = landmarks_3d[lm_id]
            R = np.array(R_all[view_id])
            t = np.array(t_all[view_id]).reshape(3,)
            K = np.array(K_all[view_id])

            projected_2d = project_point(X_world, R, t, K)
            error = np.linalg.norm(projected_2d - gt_2d)

            # 対応する部位に分類
            if lm_id in right_hand_ids:
                errors["right_hand"].append(error)
            elif lm_id in left_hand_ids:
                errors["left_hand"].append(error)
            elif lm_id in right_leg_ids:
                errors["right_leg"].append(error)
            elif lm_id in left_leg_ids:
                errors["left_leg"].append(error)
        print(f"frame_num:{lm_id}")

        # === MPJPE 出力 ===
        for part, part_errors in errors.items():
            if part_errors:
                mpjpe = np.mean(part_errors)
                print(f"{part} MPJPE: {mpjpe:.2f} cm")
            else:
                print(f"{part}: ランドマークが見つかりませんでした。")
