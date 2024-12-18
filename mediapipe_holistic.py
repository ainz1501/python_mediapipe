import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime 

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

"""
入力、グローバル変数
IMG_PATH, IMG2_PATH : 左右の画像パス
image, image2       : 左右画像
HEIGHT, WIDTH       : 画像の高さ、幅（左右同じカメラ、デバイスを使用しているため１つずつ）
K                   : カメラパラメータ（左右同じカメラ、デバイスを使用しているため１つのみ）. 単位は[px]
"""
# 画像読み込み サイズは横5712 × 縦4284
IMG_PATH = '/Users/tokudataichi/Documents/python_mediapipe/input_images/60left.jpg'
IMG2_PATH = '/Users/tokudataichi/Documents/python_mediapipe/input_images/right50_image.JPG'
image = cv2.imread(IMG_PATH)
image2 = cv2.imread(IMG2_PATH)

# 高さ、幅（同じカメラを用いるため片方の画像から取得）
HEIGHT, WIDTH, _ = image.shape

K = np.array([[4437.04178, 0, 2165.73130], [0, 4515.94693, 2783.42064], [0, 0, 1]])  

# # BGRをRGBに変換（MediaPipeがRGBを期待するため）
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

"""
関数概要
Landmark_detect(左画像, 右画像)  -> 左全身キーポイント, 右全身キーポイント, マッチングリスト
Normalized_screen_coord(左全身キーポイント, 右全身キーポイント, 画像幅, 画像高さ) -> 左画像座標系キーポイント, 右画像座標系キーポイント, マッチングリスト
Matching_landmarks(左全身キーポイント, 右全身キーポイント) -> マッチングリスト
Cam_pose_estimate(左画像座標系キーポイント, 右画像座標系キーポイント, カメラパラメータ, マッチングリスト) -> 回転行列, 並進ベクトル（左カメラから右カメラ）, マスク
Projection_mat_calc(カメラパラメータ, 回転行列, 並進ベクトル) -> 左投影行列, 右投影行列
Triangulate_3dpoint(左投影行列, 右投影行列, 左画像座標系キーポイント, 右画像座標系キーポイント, マッチングリスト) -> 3次元ランドマーク
"""
def Landmark_detect(image_left, image_right):
    """
    左右画像のランドマーク
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

    return pose_left, pose_right, matchlist

def Normalized_to_screen_coord(result, width, height):
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
    初期画像ペアの対応点からカメラポーズを推定

    Parameters:
    kp1 (list of cv2.KeyPoint): 初期画像ペアの1枚目の特徴点リスト
    kp2 (list of cv2.KeyPoint): 初期画像ペアの2枚目の特徴点リスト
    matches (list of cv2.DMatch): 初期画像ペアの対応点を表すcv2.DMatchオブジェクトのリスト
    K (numpy.ndarray): 3行3列のカメラパラメータ行列

    Returns:
    R (numpy.ndarray): 3行3列の回転行列
    t (numpy.ndarray): 3行1列の並進ベクトル
    mask_pose (numpy.ndarray): インライアを示すマスク
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
    # 左投影行列 P_left = K[I|0] = [K|0]
    Pleft = np.hstack((K, np.zeros((3, 1))))
    Pright = np.hstack((K @ R, K @ t))

    return Pleft, Pright

def Triangulate_3Dpoint(Pleft, Pright, landmark_left, landmark_right, matchlist):
    """
    カメラの投影行列を使用して特徴点の3D位置を再構築する

    Parameters:
    P1 (numpy.ndarray): 1枚目の画像のカメラ投影行列
    P2 (numpy.ndarray): 2枚目の画像のカメラ投影行列
    kp1 (list of cv2.KeyPoint): 画像1の特徴点リスト
    kp2 (list of cv2.KeyPoint): 画像2の特徴点リスト
    matches (list of cv2.DMatch): 画像間のマッチングされた特徴点のリスト
    mask (numpy.ndarray): インライアを示すマスク

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

pose_left, pose_right, matchlist = Landmark_detect(image_left=image, image_right=image2)
R, T = Cam_pose_estimate(pose_left, pose_right, matchlist, K)
Pleft, Pright = Projection_mat_calc(K, R, T)
landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right, matchlist)
print("landmark3D:","\n",landmark3D)
"""
前のやつ
"""
# 平行ステレオビジョン
def stereo_vision_parallel(pose1, pose2, width, height):
    stereo_Stime = datetime.now()
    pose_3d_landmarks = []
    right_hand_3d = []
    left_hand_3d = []
    baseline = 500.0 # 単位は[mm]
    focal_length = 26.0 # 単位は[mm]
    pixel_pitch = 0.002 #　単位は[mm]

    # ３Dジョイント位置計算
    if pose1.size > 0 and pose2.size > 0: 
        for p1, p2 in zip(pose1, pose2):
            u_l, v_l = p1[0]-width/2, p1[1]-height/2
            u_r, v_r = p2[0]-width/2, p2[1]-height/2
            joint_x = (baseline*u_l)/(u_l-u_r) # 単位は全て[mm]
            joint_y = (baseline*v_l)/(u_l-u_r)
            joint_z = (focal_length*baseline)/(pixel_pitch*(u_l-u_r)) # pixel_pitchのとこがあってないかも？

            print(f"x:{joint_x}, y:{joint_y}, z:{joint_z}")
            pose_3d_landmarks.append([joint_x, joint_y, joint_z])
        pose_3d_landmarks = np.array(pose_3d_landmarks)
    
    stereo_Etime = datetime.now()
    print(f"stereo time:{stereo_Etime - stereo_Stime}")

    return pose_3d_landmarks

# 外部パラメータ演算関数
def estimate_external_params(left, right, K):
    
    # 基本行列
    E, _ = cv2.findEssentialMat(left, right, K, cv2.RANSAC)
    # 回転行列、並進ベクトル
    _, R, T, _ = cv2.recoverPose(E, left, right)

    # SVDによるR,Tが正しいのかを判定
    print(f"R={R[:3]}")
    print(f"T={T[:3]}")
    # print(f"SVD chack:{np.cross(R,T) == E}")

    return R, T 

def estimate_initial_pose(kp1, kp2, matches, K):
    """
    初期画像ペアの対応点からカメラポーズを推定

    Parameters:
    kp1 (list of cv2.KeyPoint): 初期画像ペアの1枚目の特徴点リスト
    kp2 (list of cv2.KeyPoint): 初期画像ペアの2枚目の特徴点リスト
    matches (list of cv2.DMatch): 初期画像ペアの対応点を表すcv2.DMatchオブジェクトのリスト
    K (numpy.ndarray): 3行3列のカメラパラメータ行列

    Returns:
    R (numpy.ndarray): 3行3列の回転行列
    t (numpy.ndarray): 3行1列の並進ベクトル
    mask_pose (numpy.ndarray): インライアを示すマスク
    """
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    # 基本行列
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC)

    # カメラ姿勢の推定
    _, R, t, mask_pose = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
    return R, t, mask_pose

# ３D位置復元
def triangulate_points(P1, P2, kp1, kp2, matches):
    """
    カメラの投影行列を使用して特徴点の3D位置を再構築する

    Parameters:
    P1 (numpy.ndarray): 1枚目の画像のカメラ投影行列
    P2 (numpy.ndarray): 2枚目の画像のカメラ投影行列
    kp1 (list of cv2.KeyPoint): 画像1の特徴点リスト
    kp2 (list of cv2.KeyPoint): 画像2の特徴点リスト
    matches (list of cv2.DMatch): 画像間のマッチングされた特徴点のリスト
    mask (numpy.ndarray): インライアを示すマスク

    Returns:
    points3D (numpy.ndarray): 再構築された3Dポイントの配列
    """
    # マッチングされた特徴点を取得
    pts1 = np.float32([kp1[m] for m in matches])
    pts2 = np.float32([kp2[m] for m in matches])
    
    # 特徴点の3D位置を再構築
    points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    
    # 同次座標を3D座標に変換
    points3D = points4D[:3] / points4D[3]

    return points3D.T

def triangulate_points_withID(P_left, P_right, left, right):
    match_left = []
    match_right = []
    match_index = []

    for idx, (l, r) in enumerate(zip(left, right)):
        if not(l.size == 0 or r.size == 0): # 対応するキーポイントが左右とも存在する場合に限定
            match_left.append(l)
            match_right.append(r)
            match_index.append(idx)
    match_npleft = np.array(match_left)
    match_npright = np.array(match_right)

    points4D = cv2.triangulatePoints(P_left, P_right, match_npleft.T, match_npright.T)
    points3D =  points4D[:3] / points4D[3]
    pointsID = np.array(match_index)
    return points3D.T, pointsID

# ステレオビジョン
def stereo_vision(pose_left, pose_right, width, height):
    # 計測開始
    stereo_Stime = datetime.now()

    # 焦点距離、ピクセルピッチ
    focal_length = 26.0 # 単位は[mm]
    pixel_pitch = 0.002 #　単位は[mm]

    # 対応点が存在するキーポイントの番号
    match_index = []
    for idx, (l, r) in enumerate(zip(pose_left, pose_right)):
        if not(l.size == 0 or r.size == 0): # 対応するキーポイントが左右とも存在する場合に限定
            match_index.append(idx)
    print(f"match_indax:{len(match_index)}")

    # カメラ内部パラメータ　（仮置き）
    internal_param = np.array([[focal_length/pixel_pitch, 0, width/2],
                                [0, focal_length/pixel_pitch, height/2],
                                [0, 0, 1]])
    
    # 左カメラの外部パラメータと投影行列
    R_left = np.eye(3)
    T_left = np.zeros((3, 1))
    P_left = np.dot(internal_param, np.hstack((R_left, T_left)))
    print(f"Pl:{P_left}")

    # 右カメラの外部パラメータと投影行列
    R_right, T_right = estimate_external_params(pose_left, pose_right, internal_param)
    P_right = np.hstack((internal_param @ R_right, internal_param @ T_right))
    print(f"Pr:{P_right}")

    # 3次元ジョイントの復元
    pose_3d_coordinate, pose_3d_ID = triangulate_points(P_left, P_right, pose_left, pose_right, match_index)
    # print(f"pose coord[0][0]:{pose_3d_coordinate[0][0]}")
    # print(f"pose ID[0]:{pose_3d_ID[0]}")
    pose_3d_landmarks = np.hstack((pose_3d_ID, pose_3d_coordinate))

    # 計測終了
    stereo_Etime = datetime.now()
    print(f"stereo time:{stereo_Etime - stereo_Stime}")

    return pose_3d_landmarks

# スクリーン座標への変換（同一デバイス使用前提）
def transform2screen(result, width, height):
    scr_list = [] # カメラ座標を記録する配列を生成
    for index, landmark in enumerate(result.landmark):
        # キーポイントの値をカメラ座標系（画像中央が原点）に変換
        scr_x = landmark.x * width
        scr_y = landmark.y * height
        # リストに挿入
        scr_list.append([scr_x, scr_y]) 
    
    scr_landmarks = np.array(scr_list)

    return scr_landmarks

        
# キーポイント座標の変換処理まとめ
def transform_result(results, width, height):

    if results.pose_landmarks: # ボディ
        pose_scr_landmarks = transform2screen(results.pose_landmarks, width, height)
        # rectification_pose = stereo_rectification(pose_scr_landmarks)
        for index, scr_landmarks in enumerate(pose_scr_landmarks):
            print(f"pose_scr_landmarks[{index}].x : {scr_landmarks}")
    else:
        pose_scr_landmarks = None
        rectification_pose = None


    if results.left_hand_landmarks: # 右手
        left_hand_scr_landmarks = transform2screen(results.left_hand_landmarks, width, height)
    else:
        left_hand_scr_landmarks = None
    if results.right_hand_landmarks: # 左手
        right_hand_scr_landmarks = transform2screen(results.right_hand_landmarks, width, height)
    else:
        right_hand_scr_landmarks = None
    
    return pose_scr_landmarks, left_hand_scr_landmarks, right_hand_scr_landmarks


# メイン処理
# with mp_holistic.Holistic(static_image_mode=True) as holistic:
#     all_process_Stime = datetime.now()
#     mediapipe_Stime = datetime.now()
#     results = holistic.process(image_rgb)
#     results2 = holistic.process(image2_rgb)
#     mediapipe_Etime = datetime.now()
#     print(f"mediapipe time:{mediapipe_Etime - mediapipe_Stime}")
    
#     # 出力したキーポイントをカメラ座標系に変換(x,yの値のみ)
#     pose_scr, left_hand_scr, right_hand_scr = transform_result(results, WIDTH, HEIGHT) 
#     pose_scr2, left_hand_scr2, right_hand_scr2 = transform_result(results2, WIDTH, HEIGHT)

#     # 3Dキーポイント復元
#     pose_3d_landmarks = stereo_vision_parallel(pose_scr, pose_scr2, WIDTH, HEIGHT) 

#     all_process_Etime = datetime.now()
#     print(f"all process time:{all_process_Etime - all_process_Stime}")


# レンダリング処理
"""
レンダリング用情報
POSE_CONNECTIONS    : MediaPipeの全身ランドマークの接続情報（ボーン）
JOINT_STYLE         : 左右画像に描画するキーポイントの描画情報
BONE_STYLE          : 左右画像に描画するボーンの描画情報
"""
# 骨格情報
POSE_CONNECTIONS = mp_pose.POSE_CONNECTIONS

# レンダリング情報
JOINT_STYLE = mp_drawing.DrawingSpec(color=(0,0,255), thickness=30, circle_radius=10)
BONE_STYLE = mp_drawing.DrawingSpec(color=(200,200,0), thickness=15)

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

rendering_Stime = datetime.now()
# ２Dスケルトンレンダリング処理
if results.pose_landmarks: # 画像１のレンダリング
    print("pose through")
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                JOINT_STYLE, BONE_STYLE) 
# if results.left_hand_landmarks:
#     print("left hand through")
#     mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)
# if results.right_hand_landmarks:
#     print("right hand through")
#     mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)

if results2.pose_landmarks: # 画像２のレンダリング
    print("pose 2 through")
    mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                JOINT_STYLE, BONE_STYLE)  
# if results2.left_hand_landmarks: 
#     print("left hand 2 through")
#     mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)
# if results2.right_hand_landmarks:
#     print("right hand 2 through")
#     mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
#                                 JOINT_STYLE, BONE_STYLE)  
cv2.imshow('Holistic Result', image)
cv2.imshow('Holistic Result 2', image2)

# 2Dプロットを表示
def plot_hand_landmarks(detection_result, connection, width, height):
    x_coords = []
    y_coords = []
    # ランドマークをプロット
    for landmark in detection_result.landmark:
        x_coords.append(landmark.x*width)
        y_coords.append(landmark.y*height)
        
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

# 検出結果を2Dプロットで表示
plot_hand_landmarks(results.pose_landmarks, POSE_CONNECTIONS, WIDTH, HEIGHT)
plot_hand_landmarks(results2.pose_landmarks, POSE_CONNECTIONS, WIDTH, HEIGHT)

# 3Dスケルトンプロット生成処理
fig = plt.figure(figsize = (8, 8))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(pose_3d_landmarks[:, 0], pose_3d_landmarks[:,1],pose_3d_landmarks[:,2], s = 1, c = "blue")
set_equal_aspect(ax)
# 骨格情報からボーンを形成
for connection in POSE_CONNECTIONS:
    start_joint, end_joint = connection
    x = [pose_3d_landmarks[start_joint, 0], pose_3d_landmarks[end_joint, 0]]
    y = [pose_3d_landmarks[start_joint, 1], pose_3d_landmarks[end_joint, 1]]
    z = [pose_3d_landmarks[start_joint, 2], pose_3d_landmarks[end_joint, 2]]
    plt.plot(x, y, z, c='red', linewidth=1)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.autoscale(None)
plt.title("3D Skeleton Visualization")
rendering_Etime = datetime.now()
print(f"rendering time:{rendering_Etime - rendering_Stime}")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
# メイン終了