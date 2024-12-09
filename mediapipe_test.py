import cv2
import os
import numpy as np
import time
from scipy.optimize import least_squares
import mediapipe as mp

IMG_DIR   = "tmp_imgs/"
IMG_GENRE = "human"
IMG_PATH  = os.path.join(IMG_DIR, IMG_GENRE)+ os.sep

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def load_images(img_path):
    """
    指定されたディレクトリからグレースケール画像を読み込む

    Parameters:
    img_path (str): 画像が保存されているディレクトリのパス

    Returns:
    images (list of numpy.ndarray): 読み込まれた画像のリスト
    """
    images = []
    for filename in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

def create_feature_extractor(extractor_type):
    """
    特徴量抽出器を作成する

    Parameters:
    extractor_type (str): 特徴量抽出器のタイプ ("SIFT", "ORB", "SURF", "BRISK", "AKAZE")

    Returns:
    extractor (cv2.Feature2D): OpenCVの特徴量抽出器オブジェクト

    Raises:
    ValueError: サポートされていない特徴量抽出器タイプが指定された場合
    """
    if extractor_type == "SIFT":
        return cv2.SIFT.create()
    elif extractor_type == "ORB":
        return cv2.ORB.create()
    elif extractor_type == "SURF":
        return cv2.xfeatures2d.SURF_create()  # 要cv2.xfeatures2dのインストール
    elif extractor_type == "BRISK":
        return cv2.BRISK.create()
    elif extractor_type == "AKAZE":
        return cv2.AKAZE.create()
    else:
        raise ValueError(f"Unsupported feature extractor type: {extractor_type}")

def detect_and_compute_features(images, extractor_type="SIFT"):
    """
    画像リストに対して特徴点の検出と記述を行う。

    Parameters:
    images (list of ndarray): 入力画像のリスト
    extractor_type (str): 使用する特徴量抽出器のタイプ ("SIFT", "ORB", "SURF", "BRISK", "AKAZE")

    Returns:
    keypoints (list of list of cv2.KeyPoint): 各画像の特徴点のリスト
    descriptors (list of ndarray): 各画像の記述子のリスト
    """
    feature_extractor = create_feature_extractor(extractor_type)
    keypoints   = []
    descriptors = []
    for img in images:
        kp, des = feature_extractor.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)
    return keypoints, descriptors

def detect_and_compute_keypoint(images):
    """
    画像リストに対して特徴点の検出と記述を行う。

    Parameters:
    images (list of ndarray): 入力画像のリスト

    Returns:
    keypoints (list of list of cv2.KeyPoint): 各画像の特徴点のリスト
    descriptors (list of ndarray): 各画像の記述子のリスト
    """
    keypoints = []
    for img in images:
        # RGBカラースケールに変換 (mediapipeはRGB画像を入力に期待するため)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 高さ、幅を取得
        height, width, _ = img.shape
        # holistic起動
        with mp_holistic.Holistic(static_image_mode = True) as holistic:
            result = holistic.process(image_rgb)
            screen_points = []
            for index, landmark in enumerate(result.landmark):
                # キーポイントの値をスクリーン座標系に変換
                scr_x = landmark.x * width
                scr_y = landmark.y * height
                # リストに挿入
                screen_points.append([scr_x, scr_y])
            kp = cv2.KeyPoint()
            keypoints.append(kp)
    return keypoints

def match_features(descriptors):
    """
    画像の特徴量記述子を用いて特徴点をマッチングする

    Parameters:
    descriptors (list of numpy.ndarray): 各画像の特徴量記述子のリスト

    Returns:
    matches (list of tuple): 画像ペア間のマッチング結果のリスト
        各タプルは (img1_index, img2_index, good_matches) の形式
        - img1_index (int): 画像1のインデックス
        - img2_index (int): 画像2のインデックス
        - good_matches (list of cv2.DMatch): 画像ペア間の良好なマッチングのリスト
    """
    bf = cv2.BFMatcher()
    matches = []
    for i, des1 in enumerate(descriptors[:-1]):
        for j, des2 in enumerate(descriptors[i+1:], start=i+1):
            matches_ij = bf.knnMatch(des1, des2, k=2)
            good_matches = []
            used_trainIdx = set()
            for m, n in matches_ij:
                if m.distance < 0.8 * n.distance and m.trainIdx not in used_trainIdx:
                    good_matches.append(m)
                    used_trainIdx.add(m.trainIdx)
            matches.append((i, j, good_matches))
    return matches

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

def triangulate_points(P1, P2, kp1, kp2, matches, mask):
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
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    # マスクに基づいてインライア特徴点を選択
    pts1 = src_pts[mask.ravel() == 1].reshape(-1, 2).T
    pts2 = dst_pts[mask.ravel() == 1].reshape(-1, 2).T
    
    # 特徴点の3D位置を再構築
    points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
    
    # 同次座標を3D座標に変換
    points3D = points4D[:3] / points4D[3]

    return points3D.T

def array_in_list(array, list_of_arrays):
    return any(np.array_equal(array, item) for item in list_of_arrays)

def find_2D_3D_correspondences(kp_list, matches_list, points3D, views):
    points2D = []
    points3D_corr = []
    
    for matches in matches_list:
        img1_idx, img2_idx, good_matches = matches
        for m in good_matches:
            if m.queryIdx in views[img1_idx]:
                img_coordinate = kp_list[img2_idx][m.trainIdx].pt
                obj_coordinate = points3D[views[img1_idx][m.queryIdx]]
                if not array_in_list(img_coordinate, points2D) and not array_in_list(obj_coordinate, points3D_corr):
                    points2D.append(img_coordinate)
                    points3D_corr.append(obj_coordinate)
            elif m.trainIdx in views[img2_idx]:
                img_coordinate = kp_list[img1_idx][m.queryIdx].pt
                obj_coordinate = points3D[views[img2_idx][m.trainIdx]]
                if not array_in_list(img_coordinate, points2D) and not array_in_list(obj_coordinate, points3D_corr):
                    points2D.append(img_coordinate)
                    points3D_corr.append(obj_coordinate)
    
    return np.array(points2D), np.array(points3D_corr)

def select_initial_image_pair(matches_list):
    max_inlier = 0
    best_pair = None
    for i in range(len(matches_list)):
        idx1, idx2, matches = matches_list[i]
        if max_inlier < len(matches):
            max_inlier = len(matches)
            best_pair = (idx1, idx2, matches)
    return best_pair

def select_next_image_to_process(views, keypoints, matches, processed_images):
    """3Dポイントに最も多く対応する未処理画像を選択する。"""
    best_image_idx = None
    matches_next = None
    max_correspondences = 0

    for i in range(len(keypoints)):
        if i in processed_images:
            continue
        # 現在の画像に関連するマッチングのみを抽出
        relevant_matches = [
            (idx1, idx2, good_matches) for idx1, idx2, good_matches in matches
            if (i == idx1 and idx2 in processed_images) or (i == idx2 and idx1 in processed_images)
        ]
        # 対応点の数をカウント
        correspondences = count_correspondences(i, relevant_matches, views)
        if correspondences > max_correspondences:
            max_correspondences = correspondences
            best_image_idx = i
            matches_next = relevant_matches
    return best_image_idx, matches_next

def count_correspondences(image_idx, relevant_matches, views):
    """指定された画像に対して、既存の3Dポイントとの対応点数をカウントする。"""
    count = 0
    for idx1, idx2, good_matches in relevant_matches:
        if image_idx == idx1:
            count += sum(1 for m in good_matches if views[idx2].get(m.trainIdx) in views[idx2])
        elif image_idx == idx2:
            count += sum(1 for m in good_matches if views[idx1].get(m.queryIdx) in views[idx1])
    return count

#def choose_additional_images():

def delete_outlier(matches_list, kp_list, K):
    new_matches_list = []
    for i in range(len(matches_list)):
        idx1, idx2, matches = matches_list[i]
        kp1 = kp_list[idx1]
        kp2 = kp_list[idx2]
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
        # 基本行列を計算
        E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC)
        # マスクを使ってマッチングリストからインライアを抽出
        mask = mask.ravel().astype(bool)
        inlier_matches = [matches[j] for j in range(len(matches)) if mask[j]]
        new_matches_list.append((idx1, idx2, inlier_matches))

    return new_matches_list

def camera_params_to_rt(camera_params):
    # camera_params は長さ6の配列:
    # 前3要素はロドリゲスベクトル形式の回転情報、後3要素は並進ベクトル
    rvec = camera_params[:3]
    tvec = camera_params[3:6]
    
    # ロドリゲスベクトルから回転行列への変換
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec

def project_point(P, point3D):
    # point3Dは [x, y, z] の形式の3D座標
    # ホモジニアス座標系に変換
    point3D_homogeneous = np.append(point3D, 1)
    
    # プロジェクション行列Pを使用して3D点を投影
    projected_point_homogeneous = P @ point3D_homogeneous
    
    # ホモジニアス座標系から通常の座標系に変換（透視除算）
    projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]
    
    return projected_point

def bundle_adjustment(params, processed_images, num_points, K, views, keypoints):
    num_cameras = len(processed_images)
    def reprojection_error(params):
        camera_params = params[:num_cameras * 6].reshape((num_cameras, 6))
        points_3D_params = params[num_cameras * 6:].reshape((num_points, 3))
        error = []
        for i in range(num_cameras):
            R, t = camera_params_to_rt(camera_params[i])
            P = K @ np.hstack((R, t.reshape(3, 1)))
            idx = processed_images[i]
            for j in views[idx]:
                point3D_idx = views[idx][j]  # 3Dポイントのインデックスを取得
                projected_point = project_point(P, points_3D_params[point3D_idx])
                observed_point = keypoints[idx][j].pt  # 対応する2Dポイント
                error.extend(projected_point - np.array(observed_point))
        return error

    # レーベンバーグ・マーカート法を指定
    result = least_squares(reprojection_error, params, method='trf', max_nfev=1)
    print("trf終了")
    return result.x

def update_parameters(optimized_params, processed_images, num_points, R_total, t_total, points3D, projection_matrices, K):
    num_cameras = len(processed_images)
    camera_params_optimized = optimized_params[:num_cameras * 6].reshape((num_cameras, 6))
    points3D[:] = optimized_params[num_cameras * 6:].reshape((num_points, 3))
    for i in range(num_cameras):
        R_opt, t_opt = camera_params_to_rt(camera_params_optimized[i])
        R_total[i] = R_opt
        t_total[i] = t_opt.reshape(3, 1)
        idx = processed_images[i]
        projection_matrices[idx] = K @ np.hstack((R_opt, t_opt.reshape(3, 1)))

def incremental_sfm(images, K, extractor_type="SIFT"):
    """
    複数画像からインクリメンタルSfMを実行する

    Parameters:
    images (list of numpy.ndarray): グレースケール画像のリスト
    K (numpy.ndarray): カメラの内部パラメータ行列
    extractor_type (str): 特徴量抽出器のタイプ ("SIFT", "ORB", "SURF", "BRISK", "AKAZE")

    Returns:
    R_total (list of numpy.ndarray): 各画像の回転行列のリスト
    t_total (list of numpy.ndarray): 各画像の並進ベクトルのリスト
    points3D (list of numpy.ndarray): 再構築された3Dポイントのリスト
    """
    keypoints, descriptors = detect_and_compute_features(images, extractor_type)
    matches = match_features(descriptors)
    matches = delete_outlier(matches, keypoints, K)
    initial_pair = select_initial_image_pair(matches)
    kp1, kp2, initial_matches = keypoints[initial_pair[0]], keypoints[initial_pair[1]], initial_pair[2]
    R, t, mask = estimate_initial_pose(kp1, kp2, initial_matches, K)
    P1_initial = np.hstack((K, np.zeros((3, 1))))
    P2_initial = np.hstack((K @ R, K @ t))

    projection_matrices = {initial_pair[0]: P1_initial, initial_pair[1]: P2_initial}

    points3D = triangulate_points(P1_initial, P2_initial, kp1, kp2, initial_matches, mask)
    R_total = [np.eye(3), R]
    t_total = [np.zeros((3, 1)), t]

    views = [dict() for _ in images]
    point_idx = 0
    for i, m in enumerate(initial_matches):
        if mask[i]:
            # if views[initial_pair[1]].get(m.trainIdx, None) is None:
            views[initial_pair[0]][m.queryIdx] = point_idx
            views[initial_pair[1]][m.trainIdx] = point_idx
            point_idx += 1
    
    processed_images = [initial_pair[0], initial_pair[1]]
    """
    # Bundle adjustment after initial pair processing
    num_points = len(points3D)
    params = np.concatenate([
        np.hstack([cv2.Rodrigues(rt)[0].flatten(), tt.flatten()])
        for rt, tt in zip(R_total, t_total)
    ] + [p.flatten() for p in points3D])
    optimized_params = bundle_adjustment(params, processed_images, num_points, K, views, keypoints)
    update_parameters(optimized_params, processed_images, num_points, R_total, t_total, points3D, projection_matrices, K)
    """
    while len(processed_images) < len(images):
        next_image_idx, matches_next = select_next_image_to_process(views, keypoints, matches, processed_images)
        processed_images.append(next_image_idx)
        if next_image_idx is None:
            break  # すべての画像が処理されたか、次に処理すべき画像がない場合
        points2D, points3D_corr = find_2D_3D_correspondences(keypoints, matches_next, points3D, views)

        _, rvec, tvec, inliers = cv2.solvePnPRansac(points3D_corr, points2D, K, None, iterationsCount=10000, reprojectionError=20, confidence=0.99)
        R_new, _ = cv2.Rodrigues(rvec)
        t_new = tvec
        R_total.append(R_new)
        t_total.append(t_new)
        P_new = np.hstack((K @ R_new, K @ t_new))
        projection_matrices[next_image_idx] = P_new

        for match in matches_next:
            P1 = projection_matrices[match[0]]
            P2 = projection_matrices[match[1]]
            kp1 = keypoints[match[0]]
            kp2 = keypoints[match[1]]
            good_matches = match[2]
            new_points = []

            for m in good_matches:
                idx1 = views[match[0]].get(m.queryIdx, None)
                idx2 = views[match[1]].get(m.trainIdx, None)
                if idx1 is None and idx2 is None:
                    # まだ3Dで復元されていない新しい点のみを三角測量
                    new_points.append(m)
                elif idx1 is not None and idx2 is None:
                    # 既に復元されている3D点を再利用
                    views[match[1]][m.trainIdx] = idx1
                elif idx1 is None and idx2 is not None:
                    # 既に復元されている3D点を再利用
                    views[match[0]][m.queryIdx] = idx2
                # elif idx1 is not None and idx2 is not None:


            if new_points:
                mask_new = np.ones(len(new_points))  # 仮のマスク
                new_points3D = triangulate_points(P1, P2, kp1, kp2, new_points, mask_new)
                points3D = np.vstack((points3D, new_points3D))                
                point_idx = 0
                for m in new_points:
                    idx = len(points3D) - len(new_points3D) + point_idx
                    views[match[0]][m.queryIdx] = idx
                    views[match[1]][m.trainIdx] = idx
                    point_idx += 1

        """
        # Update after processing each new image

        num_points = len(points3D)
        params = np.concatenate([
            np.hstack([cv2.Rodrigues(rt)[0].flatten(), tt.flatten()])
            for rt, tt in zip(R_total, t_total)
        ] + [p.flatten() for p in points3D])
        optimized_params = bundle_adjustment(params, processed_images, num_points, K, views, keypoints)
        update_parameters(optimized_params, processed_images, num_points, R_total, t_total, points3D, projection_matrices, K)
        """
    return R_total, t_total, points3D


# メイン処理
if __name__ == "__main__":
    images = load_images(IMG_PATH)
    K = np.array([[2905.88, 0, 1416], [0, 2905.88, 1064], [0, 0, 1]])  # 内部カメラパラメータ
    R_total, t_total, points3D = incremental_sfm(images, K)
    
    x_points = np.array([[x, 0, 0] for x in np.arange(-0.15, 0.16, 0.01)])
    z_points = np.array([[0, 0, z] for z in np.arange(0.0, 0.41, 0.01)])
    camera_points3D = np.vstack((x_points, z_points)).reshape(-1, 3, 1)

    camera_pose_list = []  # 空のリストを初期化
    for R, t in zip(R_total, t_total):
        for points in camera_points3D:
            camera_pose_cal = R @ points + t
            camera_pose_list.append(camera_pose_cal.flatten())

    # リスト内の配列を一つのNumPy配列に結合
    camera_pose = np.vstack(camera_pose_list)

    # ファイルへの書き出し
    np.savetxt("camera_pose_tmp.xyz", camera_pose, fmt="%f %f %f")
    np.savetxt("output_tmp.xyz", points3D, fmt="%f %f %f")

    print("3Dポイントクラウドの数:", points3D.shape[0])
    print("3Dポイントクラウド:\n", points3D)
