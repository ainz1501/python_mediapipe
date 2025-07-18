import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 

# 外部パラメータ設定
ANGLE60_BASE1500 = "angle60_base1500"

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 入力
main_input_outline = 0
"""
入力、グローバル変数
IMGLEFT_PATH, IMGRIGHT_PATH : 左右の画像パス
image_left, image_right       : 左右画像
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

# 処理関数
def Landmark_detect(image_left, image_right):
    """
    左右画像のランドマークを推定し、画像座標系のそれぞれのランドマークとマッチングリストを返す

    Parameters:
    image_left (MatLike): 左カメラの画像
    image_right (MatLike): 右カメラの画像

    Returns:
    pose_left (numpy.ndarray): 33行2列の左画像の全身の画像座標系ランドマーク
    pose_right (numpy.ndarray): 33行2列の右画像の全身の画像座標系ランドマーク
    matchlist (list): 左右で規定値以上の信頼度をもつ対応するランドマークが存在することを示すマッチングリスト
    """
    # BGRスケールからRGBスケールに変換
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

    matching_list = Matching_landmarks(result_left.pose_landmarks, result_right.pose_landmarks, enable=False)

    # 推定結果を画像に描写
    annotation_image(image_left, result_left.pose_landmarks, POSE_CONNECTIONS, JOINT_STYLE, BONE_STYLE)
    annotation_image(image_right, result_right.pose_landmarks, POSE_CONNECTIONS, JOINT_STYLE, BONE_STYLE)

    return pose_left, pose_right, matching_list

def Normalized_to_screen_coord(result, width, height):
    """
    左右画像で推定したランドマークを画像座標系（スクリーン座標系）の値に変換する

    Parameters:
    result (NamedTuple): 推定したランドマークのリスト。result.landmark.x各ランドマークのx座標の値を取り出せる
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

def Matching_landmarks(landmark_left, landmark_right, enable):
    """
    左右の対応するランドマークの信頼度から、信頼度の低いランドマークを含むペアを弾くためのリストを生成する処理

    Parameter:
    landmark_left (NamedTuple): 推定した左画像のランドマーク。.landmark.visiblity で推定したランドマークに対する信頼度を取り出せる
    landmark_right (NamedTuple): 推定した右画像のランドマーク
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
    print("match_list:", matching_list)

    return matching_list

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

def Cam_pose_Preset(option):
    """
    入力されたオプションから対応した回転行列と並進ベクトルを返す関数

    Parameter:
    option: 以下の定数名を入力することで目的に応じた値を返す

    Returns:
    R: 3行3列の回転行列
    t: 3行1列の並進ベクトル
    """
    if option == ANGLE60_BASE1500: # 撮影角度60° ベースライン1500mm
        R = np.array([[0.5, 0, -(np.sqrt(3))/2.0],
              [0, 1, 0],
              [np.sqrt(3)/2.0, 0, 0.5]])
        t = np.array([[np.sqrt(3)/2.0*1500], [0], [0.5*1500]])
    
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
    
    # 特徴点の3D位置を再構築　（4行n列）
    points4D = cv2.triangulatePoints(Pleft, Pright, left_pts, right_pts)
    
    # 同次座標を3D座標に変換
    points3D = points4D[:3] / points4D[3] # 3行n列

    landmark3D = np.zeros((len(matchlist), 3))
    i = 0 # イテレーター
    for idx, match in enumerate(matchlist):
        if match == 1:
            landmark3D[idx] = points3D.T[i]
            i+=1

    return landmark3D

# 結果表示用関数
def annotation_image(image, landmarks, connections, jointstyle, bonestyle):
    """
    annotation_image(画像, ランドマーク, ボーン情報, ジョイントスタイル, ボーンスタイル)
    mediapipeが推定したランドマークを画像に描画する関数

    Parameters:
    image (MatLike): 描画する画像
    landmarks (NamedTuple): 推定したランドマークのリスト。
    connections (List of List of int): ランドマークを繋ぐボーンの連結情報。連結する2つのランドマークの数字がリスト化されている
    jointstyle (mp_drawing.DrawingSpec): ランドマークを示す点の描写情報
    bonestyle (mp_drawing.DrawingSpec): ランドマークを繋ぐ線の描写情報
    """
    mp_drawing.draw_landmarks(image, landmarks, connections, 
                            jointstyle, bonestyle)
    cv2.imshow('Holistic Result', image)
    cv2.waitKey(0)

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
    plt.show()

def plot_3Dskeleton(landmarks, connections, matchlist):
    fig = plt.figure(figsize = (8, 8))
    ax= fig.add_subplot(111, projection='3d')
    ax.scatter(landmarks[:, 0], landmarks[:,1],landmarks[:,2], s = 1, c = "blue")
    # for i in range(len(landmarks[:, 0])):
    #     ax.text(landmarks[i,0], landmarks[i,1], landmarks[i,2], str(i), dir=None, color="black", fontsize=8, ha='right', va='bottom')
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

"""
-------------------------------------------------------------------------------------------------------
"""
# メイン処理部
# ランドマーク、マッチングリスト
pose_left, pose_right, matchlist = Landmark_detect(image_left, image_right)
# 回転行列、並進ベクトル
R, T = Cam_pose_Preset(ANGLE60_BASE1500)
# 投影行列
Pleft, Pright = Projection_mat_calc(K, R, T)
# 3次元位置
landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right, matchlist)
print("landmark3D:","\n",landmark3D)

# レンダリング処理
plot_2Dskeleton(pose_left, POSE_CONNECTIONS)
plot_2Dskeleton(pose_right, POSE_CONNECTIONS)
plot_3Dskeleton(landmark3D, POSE_CONNECTIONS, matchlist)


cv2.destroyAllWindows() 