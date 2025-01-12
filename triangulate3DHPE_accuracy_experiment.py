import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 
import sys
import json
import glob
import os

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# hd00_2 13枚目 
# ボーン情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
# レンダリング情報
JOINT_STYLE = mp_drawing.DrawingSpec(color=(0,0,255), thickness=5, circle_radius=3)
BONE_STYLE = mp_drawing.DrawingSpec(color=(200,200,0), thickness=5)
mp_drawing._VISIBILITY_THRESHOLD = 0.0 # 画像に表示する際のランドマークの信用度閾値

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

def plot_3Dskeleton(landmarks, connections, matchlist, save_path):
    fig = plt.figure(figsize = (8, 8))
    ax= fig.add_subplot(111, projection='3d')
    ax.scatter(landmarks[:, 0], landmarks[:,1],landmarks[:,2], s = 1, c = "blue")
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
    plt.title("3D Skeleton Visualization")
    # 対象を正面から見た3Dプロットを画像として保存
    ax.view_init(elev=180, azim=5, roll=-90)
    fig.savefig(save_path+"3dplot.png")

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

def concatenate_images(frame_num, storage_path, save_path, show_flag):
    # 注釈前・注釈後の画像の読み込み
    original_images = (cv2.imread(storage_path+"original_L.png"), cv2.imread(storage_path+"original_R.png"))
    annotated_images = (cv2.imread(storage_path+"annotated_L.png"), cv2.imread(storage_path+"annotated_R.png"))
    HEIGHT, _, _ = original_images[0].shape # 画像高さ取得. HEIGHT=1080, WIDTH=1920

    # 3Dプロットの読み込みとリサイズ
    plot3d_image = cv2.imread(storage_path+"3dplot.png")
    plot3d_image_resize = cv2.resize(plot3d_image, (HEIGHT, HEIGHT), interpolation=cv2.INTER_CUBIC)
    # 画像を横に連結
    concatenation_left_image = cv2.hconcat([original_images[0], annotated_images[0], plot3d_image_resize])
    concatenation_right_image = cv2.hconcat([original_images[1], annotated_images[1], plot3d_image_resize])
    if show_flag:
        cv2.imshow("concatenation_image", concatenation_left_image)

    # 連結画像を保存
    if save_path is not None:
        frames_left_name = "frame_left_"+str(frame_num).zfill(4)+".jpg" # 例："frame_left_0001.jpg"
        frames_right_name = "frame_right_"+str(frame_num).zfill(4)+".jpg" # 例："frame_right_0001.jpg"
        cv2.imwrite(save_path+frames_left_name, concatenation_left_image)
        cv2.imwrite(save_path+frames_right_name, concatenation_right_image)

    return concatenation_left_image, concatenation_right_image

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
# 使用ビデオ番号 20,21番の動画は使用できない
USE_MAIN_VIDEO_NUM = 0
START_VIDEO_NUM = 22
# 使用データセット
DATASET_NAME = "171204_pose3"
# キャリブレーションファイル呼び出し
calibration_file = open("./panoptic-toolbox/"+DATASET_NAME+"/calibration_"+DATASET_NAME+".json")
parameters = json.load(calibration_file)
print("cam_params:", not(parameters is None))

for video_num in range(START_VIDEO_NUM, 31):
    print("video:",video_num)
    # 入力動画
    VIDEO1 = cv2.VideoCapture("./panoptic-toolbox/"+DATASET_NAME+"/hdVideos/hd_00_"+str(USE_MAIN_VIDEO_NUM).zfill(2)+".mp4")
    VIDEO2 = cv2.VideoCapture("./panoptic-toolbox/"+DATASET_NAME+"/hdVideos/hd_00_"+str(video_num).zfill(2)+".mp4")
    # 内部パラメータ、外部パラメータ
    param1 = parameters["cameras"][479+USE_MAIN_VIDEO_NUM] # HDカメラ00_00 [479]
    param2 = parameters["cameras"][479+video_num]
    K_left = np.array(param1["K"])
    R_left, T_left = np.array(param1["R"]), np.array(param1["t"])
    K_right = np.array(param2["K"])
    R_right, T_right = np.array(param2["R"]), np.array(param2["t"])

    # 一時的に画像を保存するフォルダのパス
    TEMPORARY_IMAGES_STORAGE_PATH = "./output_images/images_temporary_storage/"
    # 作成した動画を保存するフォルダのパス
    VIDEO_STORAGE_PATH = "./output_videos/"+DATASET_NAME+"_hdvideo_"+str(USE_MAIN_VIDEO_NUM).zfill(2)+"&"+str(video_num).zfill(2)+"/"
    if not os.path.isdir(VIDEO_STORAGE_PATH): # 指定したフォルダがなければ作成
        os.makedirs(VIDEO_STORAGE_PATH)
    # キャプチャーするフレーム間隔
    capture_rate = 100

    frame_num = 1
    concatnate_left = []
    concatnate_right = []
    while True:
        print("frame "+str(frame_num)) # 現在のキャプチャフレーム数を表示
        
        # キャプチャー処理
        for i in range(capture_rate): # capture_rateフレームごとにキャプチャ
            # ビデオキャプチャー
            ret1, img1 = VIDEO1.read()
            ret2, img2 = VIDEO2.read()
        if not (ret1 and ret2):
            print("breaked frame:", frame_num)
            break
        
        # 三角測量を用いた3Dポーズ推定(HPE)
        # 高さ、幅（同じカメラを用いるため片方の画像から取得）
        HEIGHT, WIDTH, _ = img1.shape   
        # CMU Panoptic HDカメラ 1920×1080
        # ランドマーク、マッチングリスト
        result_left, result_right = Landmark_detect(image_left=img1, image_right=img2)
        if (result_left.pose_landmarks is None) or (result_right.pose_landmarks is None):
            if result_left.pose_landmarks is None:
                cv2.imwrite(VIDEO_STORAGE_PATH+"disable_frame_L_"+str(frame_num)+".png", img1)
            if result_right.pose_landmarks is None:
                cv2.imwrite(VIDEO_STORAGE_PATH+"disable_frame_L_"+str(frame_num)+".png", img1)
            print("no pose")
        else:
            # 画像座標系への変換(同時に体のランドマークのみ使用)
            pose_left, pose_right = Normalized_to_screen_coord(result_left.pose_landmarks, result_right.pose_landmarks, WIDTH, HEIGHT)
            # print("left=","\n",pose_left)
            # print("right=","\n",pose_right)
            # 投影行列を計算
            Pleft, Pright = Projection_mat_calc(K_left, R_left, T_left, K_right, R_right, T_right)
            # 3次元位置を復元
            landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right)
            # print("landmark3D:","\n",landmark3D)

            # 比較用画像を作成
            annotate_image(img1, result_left.pose_landmarks, img2, result_right.pose_landmarks, 
                                                                    POSE_CONNECTIONS, JOINT_STYLE, BONE_STYLE)
            matching_list = Matching_landmarks(result_left.pose_landmarks, result_right.pose_landmarks, enable=False)
            fig, ax = plot_3Dskeleton(landmark3D, POSE_CONNECTIONS, matching_list, save_path=TEMPORARY_IMAGES_STORAGE_PATH)
            # 比較用画像を横に連結する
            out_left, out_right = concatenate_images(frame_num, storage_path=TEMPORARY_IMAGES_STORAGE_PATH, save_path=None, show_flag=False)
            concatnate_left.append(out_left)
            concatnate_right.append(out_right)
            
            print("out num ",len(concatnate_left))
            # cv2.waitKey(0)
            cv2.destroyAllWindows() 
        frame_num += 1

    # 比較用動画の作成
    create_video_from_images(concatnate_left, concatnate_right, frame_rate=5, save_path=VIDEO_STORAGE_PATH)