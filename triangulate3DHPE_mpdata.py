import cv2
import mediapipe as mp
import numpy as np
import json
import os

# MediaPipe Holisticモジュールを初期化
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

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

# 設定パラメータ
START_FRAME = 137
END_FRAME = 9056
FRAME_INTERVAL = 100
NUM_VIEWS = 31  # 0〜30
NUM_LANDMARKS = 33
EXCLUDED_VIEWS = [20, 21]

# ビデオキャプチャを用意
video_caps = {
    view_id: cv2.VideoCapture(f"./panoptic-toolbox/{DATASET_NAME}/hdVideos/hd_00_{str(view_id).zfill(2)}.mp4")
    for view_id in range(31) if view_id not in EXCLUDED_VIEWS
}

#jsonファイル保存フォルダ作成
if not os.path.isdir(f"./outputs/best_landmarks"): # 指定したフォルダがなければ作成
    os.makedirs(f"./outputs/best_landmarks")

# 出力データ
result = {}

# 使用ビュー番号
views_to_use = [i for i in range(31) if i not in EXCLUDED_VIEWS]

# フレームごとに処理
for frame_id in range(START_FRAME, END_FRAME + 1, FRAME_INTERVAL):
# for frame_id in range(START_FRAME, START_FRAME+1, FRAME_INTERVAL): # テスト用
    result[frame_id] = {}

    # 各ランドマークについて最も信頼度が高かったビュー情報を格納
    best = {i: {"view": None, "x": None, "y": None, "visibility": -1.0} for i in range(NUM_LANDMARKS)}

    for view_id in views_to_use:
        cap = video_caps[view_id]
        if not cap.isOpened():
            continue

        # 対象フレームにジャンプ
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret:
            continue

        # MediaPipe Pose に通す
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            h, w = frame.shape[:2]
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                if lm.visibility > best[idx]["visibility"]:
                    best[idx] = {
                        "view": view_id,
                        "x": lm.x * w,
                        "y": lm.y * h,
                        "visibility": lm.visibility
                    }

    # このフレームの結果を保存
    result[frame_id] = best

# JSONファイルとして保存
with open(f"./outputs/best_landmarks/best_landmarks_{str(frame_id).zfill(4)}.json", "w") as f:
    json.dump(result, f, indent=2)

pose.close()
print("完了：best_landmarks_by_view.json に保存しました。")
