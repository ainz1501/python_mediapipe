import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 
import sys
import json
import time

# 使用データセット
DATASET_NAME = "171204_pose3"
# 使用ビデオ番号
USE_VIDEO_NUM = 25

# キャリブレーションファイル呼び出し
calibration_file = open("./panoptic-toolbox/"+DATASET_NAME+"/calibration_"+DATASET_NAME+".json")
parameters = json.load(calibration_file)
print("cam_params:", not(parameters is None))

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 入力動画
VIDEO1 = cv2.VideoCapture("./panoptic-toolbox/"+DATASET_NAME+"/hdVideos/hd_00_00.mp4")
VIDEO2 = cv2.VideoCapture("./panoptic-toolbox/"+DATASET_NAME+"/hdVideos/hd_00_"+str(USE_VIDEO_NUM)+".mp4")
# 内部パラメータ、外部パラメータ
param1 = parameters["cameras"][479] # HDカメラ00_00 [479]
param2 = parameters["cameras"][479+USE_VIDEO_NUM]
K_left = np.array(param1["K"])
R_left, T_left = np.array(param1["R"]), np.array(param1["t"])
K_right = np.array(param2["K"])
R_right, T_right = np.array(param2["R"]), np.array(param2["t"])


# レンダリング用入力
# 一時的に画像を保存するフォルダパス
TEMPORARY_IMAGES_STORAGE_PATH = "./output_images/images_temporary_storage/"
VIDEO_STORAGE_PATH = "./output_videos/"+DATASET_NAME+"_hdvideo_00&25/"
# ボーン情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
# レンダリング情報
JOINT_STYLE = mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=3)
BONE_STYLE = mp_drawing.DrawingSpec(color=(200,200,0), thickness=5)

# より正確な時間の計測のため、prof_counter関数自体の処理時間を確かめる
# 処理関数
def Landmark_detect(image_left, image_right):
    """
    左右画像それぞれの対象の手、顔、全身の3Dポーズを推定する

    Parameters:
    image_left (MatLike)        : 左カメラの画像
    image_right (MatLike)       : 右カメラの画像

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

# def Cam_pose_estimate(landmark_left, landmark_right, matchlist, K):
    """
    左右画像の対応ランドマークからカメラポーズを推定

    Parameters:
    landmark_left (numpy.ndarray)   : 左画像のランドマーク
    landmark_right (numpy.ndarray)  : 右画像のランドマーク
    matchlist (list of int)         : 画像ペアで対応するランドマークの有無を示すリスト
    K (numpy.ndarray)               : 3行3列のカメラパラメータ行列

    Returns:
    R (numpy.ndarray)               : 3行3列の回転行列
    t (numpy.ndarray)               : 3行1列の並進ベクトル
    """
    src_list = []
    dst_list = []
    for idx, match in enumerate(matchlist):
        if match == 1:
            src_list.append(landmark_left[idx])
            dst_list.append(landmark_right[idx])
    src_pts = np.array(src_list)
    dst_pts = np.array(dst_list)
    print("src_pts:","\n",src_pts)
    print("dst_pts:","\n",dst_pts)
    

    # 基本行列
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC)

    # カメラ姿勢の推定
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
    
    print("E:",E)
    print("R:",R)
    print("t:",t)   
    return R, t

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
    points3D (numpy.ndarray)        : 再構築されたn行3列の3Dポイントの配列
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

    if (result_left.pose_landmarks is None) or (result_right.pose_landmarks is None):
        landmark3D = None
    else:
        # 画像座標系への変換(同時に体のランドマークのみ使用)
        pose_left, pose_right = Normalized_to_screen_coord(result_left.pose_landmarks, result_right.pose_landmarks, WIDTH, HEIGHT)
        # 投影行列を計算
        Pleft, Pright = Projection_mat_calc(K_l, R_l, T_l, K_r, R_r, T_r)
        # 3次元位置を復元
        landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right)

    return landmark3D

def mean_processing_time_calc(time_list, through_list):
    t = []
    for index, flag in enumerate(through_list):
        if flag == 1:
            t.append(time_list[index])
    time_sum = sum(t)
    list_len = len(t)
    average = sum(t)/len(t)

    return average

    
"""
-------------------------------------------------------------------------------------------------------
"""
# メイン処理部
frame_num = 1
capture_rate = 1000
test_times = 10000

while True:
    print("frame "+str(frame_num)) # 現在のキャプチャフレーム数を表示
    
    # キャプチャー処理
    for i in range(capture_rate): # cupture_rateの値ごとにキャプチャ
        # ビデオキャプチャー
        ret1, img1 = VIDEO1.read()
        ret2, img2 = VIDEO2.read()
    if not (ret1 and ret2):
        print("breaked frame:", frame_num)
        break
    

    # 三角測量を用いた3Dポーズ推定(HPE)
    # 高さ、幅（同じカメラを用いるため片方の画像から取得）
    HEIGHT, WIDTH, _ = img1.shape   
    # ランドマークを推定
    result_left, result_right = Landmark_detect(image_left=img1, image_right=img2)

    if (result_left.pose_landmarks is None) or (result_right.pose_landmarks is None): # 両画像が推定できる画像かを判定
        print("no pose")
    else:
        # # 相対的3Dランドマーク推定
        # t1 = time.perf_counter()
        # for i in range(test_times): 
        #     _, _ = Landmark_detect(img1, img2)
        #     print("times:",i)
        # t2 = time.perf_counter()
        # # Landmark_detect_time = t2-t1

        img1_RGB = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2_RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # # 相対的3Dランドマーク推定（左画像）
        # t1 = time.perf_counter()
        # for i in range(test_times): 
        #     with mp_holistic.Holistic(static_image_mode=True) as holistic:
        #         result_left = holistic.process(img1_RGB)
        #     print("times:",i)
        # t2 = time.perf_counter()
        # Landmark_detect_left_time = t2-t1
        
        # # 相対的3Dランドマーク推定（右画像）
        # t1 = time.perf_counter()
        # for i in range(test_times): 
        #     with mp_holistic.Holistic(static_image_mode=True) as holistic:
        #         result_left = holistic.process(img1_RGB)
        #     print("times:",i)
        # t2 = time.perf_counter()
        # Landmark_detect_right_time = t2-t1

        # # 相対的3Dランドマーク推定（左右画像）
        # t1 = time.perf_counter()
        # for i in range(test_times): 
        #     with mp_holistic.Holistic(static_image_mode=True) as holistic:
        #         result_left = holistic.process(img1_RGB)
        #         result_right = holistic.process(img2_RGB)
        #     print("times:",i)
        # t2 = time.perf_counter()
        # Landmark_detect_both_time = t2-t1

        # # 画像座標系への変換(同時に体のランドマークのみ使用)
        # t1 = time.perf_counter()
        # for i in range(test_times):
        #     pose_left, pose_right = Normalized_to_screen_coord(result_left.pose_landmarks, result_right.pose_landmarks, WIDTH, HEIGHT)
        #     print("times:",i)
        # t2 = time.perf_counter()
        # Normalized_to_screen_coord_time = t2-t1

        # # 投影行列を計算
        # t1 = time.perf_counter()
        # for i in range(test_times):
        #     Pleft, Pright = Projection_mat_calc(K_left, R_left, T_left, K_right, R_right, T_right)
        #     print("times:",i)
        # t2 = time.perf_counter()
        # Projection_mat_calc_time = t2-t1

        # # 3次元位置を復元
        # t1 = time.perf_counter()
        # for i in range(test_times):
        #     landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right)        
        #     print("times:",i)
        # t2 = time.perf_counter()
        # Triangulate_3Dpoint_time = t2-t1

        # # 全体の処理
        # t1 = time.perf_counter()
        # for i in range(test_times):
        #     landmark3D = Triangulate3DHPE(img1, img2, K_left, R_left, T_left, K_right, R_right, T_right)
        #     print("times:",i)
        # t2 = time.perf_counter()
        # entire_time = t2-t1

        # ループ回数表示
        t1 = time.perf_counter()
        for i in range(test_times):
            print("times:",i)
        t2 = time.perf_counter()
        print_times_time = t2-t1

        break
    frame_num += 1

# entire_ave = mean_processing_time_calc(entire_time_list, through_list)
# Landmark_detect_ave = mean_processing_time_calc(Landmark_detect_time_list, through_list)
# Landmark_detect_left_ave = mean_processing_time_calc(Landmark_detect_left_time_list, through_list)
# Landmark_detect_right_ave = mean_processing_time_calc(Landmark_detect_right_time_list, through_list)
# Normalized_to_screen_coord_ave = sum(Normalized_to_screen_coord_time_list)/len(Normalized_to_screen_coord_time_list)
# Projection_mat_calc_ave = sum(Projection_mat_calc_time_list)/len(Projection_mat_calc_time_list)
# Triangulate_3Dpoint_ave = sum(Triangulate_3Dpoint_time_list)/len(Triangulate_3Dpoint_time_list)

# print("entire:", entire_time)
# print("Landmark_detect:", Landmark_detect_time)
# print("Normalized_to_screen_coord:", Normalized_to_screen_coord_time)
# print("Projection_mat_calc:", Projection_mat_calc_time)
# print("Triangulate_3Dpoint:", Triangulate_3Dpoint_time)
print("print times:", print_times_time)

# print("entire culc:",Landmark_detect_time+Normalized_to_screen_coord_time+Projection_mat_calc_time+Triangulate_3Dpoint_time)
# print("Landmark_detect left-image:", Landmark_detect_left_time)
# print("Landmark_detect right-image:", Landmark_detect_right_time)
# print("Landmark_detect both-image:", Landmark_detect_both_time)

# print("Landmark_detect per:", "{:.3f}".format(Landmark_detect_time/entire_time*100), "%")
# print("Normalized_to_screen_coord per:", "{:.3f}".format(Normalized_to_screen_coord_time/entire_time*100), "%")
# print("Projection_mat_calc per:", "{:.3f}".format(Projection_mat_calc_time/entire_time*100), "%")
# print("Triangulate_3Dpoint per:", "{:.3f}".format(Triangulate_3Dpoint_time/entire_time*100), "%")

# print("entire fps:", "{:.4f}".format(10000/entire_time)) # 1/(entire_time/10000)

print("finish")