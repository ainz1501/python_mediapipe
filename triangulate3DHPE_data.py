import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from typing import List
"""
実験で使用するフレームを抜き出して保存する
推定した3Dランドマークをリスト化して保存する
"""
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

# ボーン情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS
# レンダリング情報
JOINT_STYLE = mp_drawing.DrawingSpec(color=(255,0,0), thickness=5, circle_radius=3)
BONE_STYLE = mp_drawing.DrawingSpec(color=(200,200,0), thickness=5)
mp_drawing._VISIBILITY_THRESHOLD = 0.0 # 画像に表示する際のランドマークの信用度閾値  

"""
Landmark_detect：2Dランドマークの推定
Normalized_to_screen_coord：画像座標系への変換
Matching_landmarks：閾値以上の信頼度をもつ対応ランドマークを特定するリストを作成
Projection_mat_calc：透視投影行列の計算
Triangulate_3Dpoint：三角測量を行う
Triangulate3DHPE：上記の処理をまとめて関数化したもの

annotate_image：元画像にMediaPipeが推定したランドマークを描画する
plot_2Dskeleton：推定した2Dランドマークを2Dプロット上にそれぞれ描画する
plot_3Dskeleton：推定した3Dランドマークを3Dプロット上に描画する
set_equal_aspect：3Dプロットの各軸の幅を揃える
concatenate_images：入力画像と注釈後の画像を連結する
create_video_from_images：左右それぞれの比較用動画を作成
"""
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

    Returns
    -------
    List[int]
        推定結果を保存したフレーム番号のリスト。
    """
    dataset_root = os.path.join(panoptic_root, dataset_name)
    video1_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video1_num).zfill(2)}.mp4")
    video2_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video2_num).zfill(2)}.mp4")
    calibration_path = os.path.join(dataset_root, f"calibration_{dataset_name}.json")

    input1_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}")
    input2_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video2_num).zfill(2)}")
    video_storage_path = os.path.join(videos_root, f"{dataset_name}_hdvideo{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")
    output_landmarks_path = os.path.join(landmarks_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")

    os.makedirs(video_storage_path, exist_ok=True)
    os.makedirs(input1_image_path, exist_ok=True)
    os.makedirs(input2_image_path, exist_ok=True)
    os.makedirs(output_landmarks_path, exist_ok=True)
    os.makedirs(temp_storage_path, exist_ok=True)

    with open(calibration_path) as calibration_file:
        parameters = json.load(calibration_file)
    print("cam_params:", parameters is not None)

    param1 = parameters["cameras"][479 + video1_num]
    param2 = parameters["cameras"][479 + video2_num]
    K_left = np.array(param1["K"])
    R_left, T_left = np.array(param1["R"]), np.array(param1["t"])
    K_right = np.array(param2["K"])
    R_right, T_right = np.array(param2["R"]), np.array(param2["t"])

    video1 = cv2.VideoCapture(video1_path)
    video2 = cv2.VideoCapture(video2_path)

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
    plt.savefig()
    plt.show()

def plot_3Dskeleton(landmarks, connections, save_path):
    fig = plt.figure(figsize = (8, 8))
    ax= fig.add_subplot(111, projection='3d')
    ax.scatter(landmarks[:, 0], landmarks[:,1],landmarks[:,2], s = 1, c = "blue")
    set_equal_aspect(ax)
    # 骨格情報からボーンを形成
    for connection in connections:
        start_joint, end_joint = connection
        x = [landmarks[start_joint, 0], landmarks[end_joint, 0]]
        y = [landmarks[start_joint, 1], landmarks[end_joint, 1]]
        z = [landmarks[start_joint, 2], landmarks[end_joint, 2]]
        plt.plot(x, y, z, c='red', linewidth=1)
            
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.title("3D Skeleton Visualization")
    # 対象を正面から大体hd_00カメラの位置から見た画像を保存
    ax.view_init(elev=180, azim=5, roll=-90)
    # fig.savefig(save_path+"3dplot.png")
    fig.show()
    
    plt.close()

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

if __name__ == "__main__":
    run_triangulate3dhpe_data(capture_rate=30)
