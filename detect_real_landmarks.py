import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 入力
main_input_outline = 0
"""
入力、グローバル変数
IMG_PATH, IMG2_PATH : 左右の画像パス
image, image2       : 左右画像
HEIGHT, WIDTH       : 画像の高さ、幅（左右同じカメラ、デバイスを使用しているため１つずつ）
K                   : カメラパラメータ（左右同じカメラ、デバイスを使用しているため１つのみ）. 単位は[px]
"""
# 画像読み込み サイズは横5712 × 縦4284
IMGLEFT_PATH = '/Users/tokudataichi/Documents/python_mediapipe/input_images/60left.jpg'
IMGRIGHT_PATH = '/Users/tokudataichi/Documents/python_mediapipe/input_images/60right.JPG'
image_left = cv2.imread(IMGLEFT_PATH)
image_right = cv2.imread(IMGRIGHT_PATH)

# 高さ、幅（同じカメラを用いるため片方の画像から取得）
HEIGHT, WIDTH, _ = image_left.shape

K = np.array([[4437.04178, 0, 2165.73130], [0, 4515.94693, 2783.42064], [0, 0, 1]])  

# レンダリング用入力
rendering_input = 0
"""
レンダリング用情報
POSE_CONNECTIONS    : MediaPipeの全身ランドマークの接続情報（ボーン）
JOINT_STYLE         : 左右画像に描画するキーポイントの描画情報
BONE_STYLE          : 左右画像に描画するボーンの描画情報
"""
# ボーン情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# レンダリング情報
JOINT_STYLE = mp_drawing.DrawingSpec(color=(0,0,255), thickness=30, circle_radius=10)
BONE_STYLE = mp_drawing.DrawingSpec(color=(200,200,0), thickness=15)

function_outline = 0
"""
関数概要
Landmark_detect(左画像, 右画像)  -> 左画像座標系キーポイント, 右画像座標系キーポイント, マッチングリスト
    ・Normalized_screen_coord(左全身キーポイント, 右全身キーポイント, 画像幅, 画像高さ) -> 左画像座標系キーポイント, 右画像座標系キーポイント, マッチングリスト
    ・Matching_landmarks(左全身キーポイント, 右全身キーポイント) -> マッチングリスト
Cam_pose_estimate(左画像座標系キーポイント, 右画像座標系キーポイント, カメラパラメータ, マッチングリスト) -> 回転行列, 並進ベクトル（左カメラから右カメラ）, マスク
Projection_mat_calc(カメラパラメータ, 回転行列, 並進ベクトル) -> 左投影行列, 右投影行列
Triangulate_3dpoint(左投影行列, 右投影行列, 左画像座標系キーポイント, 右画像座標系キーポイント, マッチングリスト) -> 3次元ランドマーク
"""
def Landmark_detect(image_left, image_right):
    """
    左右画像のランドマークを推定し、画像座標系のそれぞれのランドマークとマッチングリストを返す

    Parameters:
    image_left (MatLike): 左カメラの画像
    image_right (MatLike): 右カメラの画像

    Returns:
    pose_left (numpy.ndarray): 33行2列の左画像の全身の画像座標系ランドマーク
    pose_right (numpy.ndarray): 33行2列の右画像の全身の画像座標系ランドマーク
    matchlist (list): 左右で対応するランドマークが存在することを示すマッチングリスト（全身に対して行っているため要素数は33個）
    """
    # RGBスケールに変換
    image_leftRGB = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_rightRGB = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

    # ランドマーク推定 静止画 => static_image_mode=True 動画 => 
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        result_left = holistic.process(image_leftRGB)
        result_right = holistic.process(image_rightRGB)

    pose_left = Normalized_to_screen_coord(result_left.pose_landmarks, WIDTH, HEIGHT)
    pose_right = Normalized_to_screen_coord(result_right.pose_landmarks, WIDTH, HEIGHT)
    print("left=","\n",pose_left)
    print("right=","\n",pose_right)

    matchlist = Matching_landmarks(pose_left, pose_right)

    # 推定結果を画像に描写
    annotation_image(image_left, result_left.pose_landmarks, POSE_CONNECTIONS, JOINT_STYLE, BONE_STYLE)
    annotation_image(image_right, result_right.pose_landmarks, POSE_CONNECTIONS, JOINT_STYLE, BONE_STYLE)

    return pose_left, pose_right, matchlist

def annotation_image(image, landmarks, connections, jointstyle, bonestyle):
    mp_drawing.draw_landmarks(image, landmarks, connections, 
                            jointstyle, bonestyle)
    cv2.imshow('Holistic Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

def Normalized_to_screen_coord(result, width, height):
    """
    左右画像で推定したランドマークを画像座標系（スクリーン座標系）の値に変換する

    Parameters:
    result (Optional_List): 推定したランドマークのリスト
    width (int): 画像の幅
    height (int): 画像の高さ

    Returns:
    scr_landmarks (numpy.ndarray): n行2列の画像座標系ランドマーク (全身なのでn=33)
    """
    scr_list = [] # カメラ座標を記録する配列を生成
    for index, landmark in enumerate(result.landmark):
        # キーポイントの値をカメラ座標系（画像中央が原点）に変換
        scr_x = landmark.x * width
        scr_y = landmark.y * height
        # リストに挿入
        scr_list.append([scr_x, scr_y]) 
    
    scr_landmarks = np.array(scr_list)

    return scr_landmarks

def Matching_landmarks(landmark_left, landmark_right):
    # 対応点が存在するキーポイントの番号
    matchlist = []
    for l, r in zip(landmark_left, landmark_right):
        if not(l.size == 0 or r.size == 0): # 対応するキーポイントが左右とも存在する場合に限定
            matchlist.append(1)
        else:
            matchlist.append(0)
    print("match_list:", matchlist)

    return matchlist

def Cam_pose_estimate(landmark_left, landmark_right, matchlist, K):
    """
    左右画像の対応ランドマークからカメラポーズを推定

    Parameters:
    landmark_left (numpy.ndarray): 左画像のランドマーク
    landmark_right (numpy.ndarray): 右画像のランドマーク
    matchlist (list of int): 画像ペアで対応するランドマークの有無を示すリスト
    K (numpy.ndarray): 3行3列のカメラパラメータ行列

    Returns:
    R (numpy.ndarray): 3行3列の回転行列
    t (numpy.ndarray): 3行1列の並進ベクトル
    """
    src_list = []
    dst_list = []
    for idx, match in enumerate(matchlist):
        if match == 1:
            src_list.append(landmark_left[idx])
            dst_list.append(landmark_right[idx])
    src_pts = np.array(src_list)
    dst_pts = np.array(dst_list)

    # 基本行列
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC)

    # カメラ姿勢の推定
    _, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
    print("mask:",mask)
    return R, t

def Projection_mat_calc(K, R, t):
    """
    左右カメラの投影行列を生成する

    Parameters:
    K (numpy.ndarray): 3行3列のカメラパラメータ
    R (numpy.ndarray): 3行3列の回転行列
    t (numpy.ndarray): 3行1列の並進ベクトル

    Returns:
    Pleft (numpy.ndarray): 左画像のカメラ投影行列 (3*4)
    Pright (numpy.ndarray): 右画像のカメラ投影行列 (3*4)
    """
    # 左カメラをワールド座標とするため、左投影行列 P_left = K[I|0] = [K|0]　となる
    Pleft = np.hstack((K, np.zeros((3, 1))))
    Pright = np.hstack((K @ R, K @ t))

    return Pleft, Pright

def Triangulate_3Dpoint(Pleft, Pright, landmark_left, landmark_right, matchlist):
    """
    カメラの投影行列を使用して特徴点の3D位置を再構築する

    Parameters:
    Pleft (numpy.ndarray): 左画像のカメラ投影行列 (3*4)
    Pright (numpy.ndarray): 右画像のカメラ投影行列 (3*4)
    landmark_left (numpy.ndarray): n行2列の左画像のランドマーク（画像座標系）
    landmark_right (numpy.ndarray): n行2列右画像のランドマーク（画像座標系）(全身なのでn=33)
    matches (list of int): 画像間で対応するランドマークの有無を示すリスト

    Returns:
    points3D (numpy.ndarray): 再構築された3Dポイントの配列
    """
    # マッチングされた特徴点を取得
    left_list = []
    right_list = []
    for idx, match in enumerate(matchlist):
        if match == 1:
            left_list.append(landmark_left[idx])
            right_list.append(landmark_right[idx])
    left_pts = np.array(left_list).T
    right_pts = np.array(right_list).T
    
    # 特徴点の3D位置を再構築
    points4D = cv2.triangulatePoints(Pleft, Pright, left_pts, right_pts)
    
    # 同次座標を3D座標に変換
    points3D = points4D[:3] / points4D[3]

    return points3D.T

# 結果表示用関数
rendering_function_outline = 0
"""
レンダリング用関数概要
annotation_image(画像, ランドマーク, ボーン情報, ジョイントスタイル, ボーンスタイル)
plot_2Dskeleton(画像座標系ランドマーク, ボーン情報)
plot_3Dskeleton(3次元ランドマーク, ボーン情報, )
"""

def plot_2Dskeleton(landmarks, connection):
    x_coords = []
    y_coords = []
    # ランドマークをプロット
    for landmark in landmarks:
        x_coords.append(landmark[0])
        y_coords.append(landmark[1])
        
    # キーポイントを描画
    plt.scatter(x_coords, y_coords, color='red')

    # ボーンを描画
    for start_idx, end_idx in connection:
        plt.plot([x_coords[start_idx], x_coords[end_idx]],
                    [y_coords[start_idx], y_coords[end_idx]],
                    color='blue')
        
    plt.gca().invert_yaxis()  # Y座標を反転して画像と一致させる
    plt.axis('equal')         # アスペクト比を正確にする
    plt.show()

def plot_3Dskeleton(landmarks, connections):
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
    plt.autoscale(None)
    plt.title("3D Skeleton Visualization")
    plt.show()

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

# メイン処理部

# ランドマーク、マッチングリスト
pose_left, pose_right, matchlist = Landmark_detect(image_left, image_right)
# 回転行列、並進ベクトル
R, T = Cam_pose_estimate(pose_left, pose_right, matchlist, K)
# 投影行列
Pleft, Pright = Projection_mat_calc(K, R, T)
# 3次元位置
landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right, matchlist)
print("landmark3D:","\n",landmark3D)
"""
-------------------------------------------------------------------------------------------------------
"""

# レンダリング処理
plot_2Dskeleton(pose_left, POSE_CONNECTIONS)
plot_2Dskeleton(pose_right, POSE_CONNECTIONS)
plot_3Dskeleton(landmark3D, POSE_CONNECTIONS)