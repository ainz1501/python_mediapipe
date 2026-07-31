"""
MediaPipe と CMU Panoptic Dataset のカメラパラメータを使って、2枚の画像から
人体ランドマークの3D座標を三角測量するためのユーティリティモジュール。

主な使い方:
    camera = Load_camera_parameters()
    result_left, result_right, points3d = Triangulate3DHPE(
        img_left,
        img_right,
        camera["K_left"],
        camera["R_left"],
        camera["T_left"],
        camera["K_right"],
        camera["R_right"],
        camera["T_right"],
    )

OpenCV 画像は BGR、MediaPipe 入力は内部で RGB に変換する。
3Dランドマークは原則として shape が (N, 3) の numpy.ndarray で扱う。
"""

import cv2
import numpy as np
from datetime import datetime 
import sys
import json
import glob
import os
from typing import Any, List, Optional, Sequence

ImageArray = np.ndarray
Landmark2DArray = np.ndarray
Landmark3DArray = np.ndarray
CameraParameterDict = dict[str, Any]
PanopticPathDict = dict[str, str]

"""
結果比較用プログラム
入力画像2枚、3Dプロットを表示
"""
    # if test_mode:
    #     test_root = "./testIO"
    #     inputs_root = os.path.join(test_root, "test_sys_inputs")
    #     videos_root = os.path.join(test_root, "test_sys_outputs", "output_videos")
    #     landmarks_root = os.path.join(test_root, "test_sys_outputs", "output_landmarks")
    #     temp_storage_path = os.path.join(test_root, "test_sys_outputs", "images_temporary_storage")
        
    #     # Create test directories
    #     os.makedirs(inputs_root, exist_ok=True)
    #     os.makedirs(videos_root, exist_ok=True)
    #     os.makedirs(landmarks_root, exist_ok=True)
    #     os.makedirs(temp_storage_path, exist_ok=True)

# デフォルト設定値
#: フレームを何枚おきに処理するかを示すデフォルト値。
DEFAULT_CAPTURE_RATE: int = 100
#: Panoptic Dataset のランドマーク JSON が存在し始めるデフォルトフレーム番号。
DEFAULT_FIRST_FRAME_NUM: int = 137
#: デフォルトで使用する Panoptic Dataset のシーケンス名。
DEFAULT_DATASET_NAME: str = "171204_pose3"
#: デフォルトの左カメラ番号。`hd_00_18.mp4` に対応する。
DEFAULT_VIDEO1_NUM: int = 18
#: デフォルトの右カメラ番号。`hd_00_23.mp4` に対応する。
DEFAULT_VIDEO2_NUM: int = 23
#: デフォルトの処理終了フレーム番号。この値は処理対象に含まない。
DEFAULT_END_FRAME_NUM: int = 9056
#: 注釈画像などの一時ファイルを保存するデフォルトディレクトリ。
DEFAULT_TEMPORARY_IMAGES_STORAGE_PATH: str = "./output_images/images_temporary_storage/"
#: Panoptic Dataset が置かれているデフォルトのルートディレクトリ。
DEFAULT_PANOPTIC_ROOT: str = "./panoptic-toolbox"
#: 抽出した入力画像を保存するデフォルトのルートディレクトリ。
DEFAULT_INPUTS_ROOT: str = "./sys_inputs/input_images"
#: 比較用動画を保存するデフォルトのルートディレクトリ。
DEFAULT_VIDEOS_ROOT: str = "./sys_outputs/output_videos"
#: 推定ランドマーク JSON を保存するデフォルトのルートディレクトリ。
DEFAULT_LANDMARKS_ROOT: str = "./sys_outputs/output_landmarks"
TEMPORARY_IMAGES_STORAGE_PATH: str = DEFAULT_TEMPORARY_IMAGES_STORAGE_PATH

# 別データ間で同じ箇所を示す対応ランドマーク番号リスト[正解データ, 推定データ]
#: Body の `[GTランドマーク番号, MediaPipeランドマーク番号]` 対応表。
BODY_MAP: list[list[int]] = [[1, 0],    # Nose
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

#: 右手の `[GTランドマーク番号, MediaPipeランドマーク番号]` 対応表。
RIGHT_HAND_MAP: list[list[int]] = [[0, 16],  # Wrist
            [17, 18],   # pinky
            [5, 20],    # index
            [2, 22]]    # thumb

#: 左手の `[GTランドマーク番号, MediaPipeランドマーク番号]` 対応表。
LEFT_HAND_MAP: list[list[int]] = [[0,15],    # Wrist
                 [17, 17],  # pinky
                 [5, 19],   # index
                 [2, 21]]   # thumb

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

# キャプチャーするフレーム
CAPTURE_RATE: int = 100
FIRST_FRAME_NUM: int = 837 # データセットに含まれるランドマークデータが137フレーム目以降からファイルが存在するため
"""
使用できないフレームについて
下記のフレームは例外となるデータであるため、次のフレームをセットして再度プログラムを実行する．
初期値 137
8337: 手のデータが右手しかなく、ほとんどのランドマークがエラーを示している
"""

# 使用データ指定
DATASET_NAME: str = "171204_pose3"
GT_DATA_FOLDER_PATH: str = "./panoptic-toolbox/"+DATASET_NAME+"/hdPose3d_stage1_coco19/"
GT_HAND_DATA_FOLDER_PATH: str = "./panoptic-toolbox/"+DATASET_NAME+"/hdHand3d/"
VIDEO1_NUM: int = 18
VIDEO2_NUM: int = 23
END_FRAME_NUM: int = 9056

# パスまとめ
TEMPORARY_IMAGES_STORAGE_PATH: str = "./output_images/images_temporary_storage/" # 一時的に画像を保存するフォルダのパス
INPUT1_IMAGE_PATH: str = "./inputs/input_images/"+DATASET_NAME+"_cam"+str(VIDEO1_NUM).zfill(2)+"/"
INPUT2_IMAGE_PATH: str = "./inputs/input_images/"+DATASET_NAME+"_cam"+str(VIDEO2_NUM).zfill(2)+"/"
VIDEO_STORAGE_PATH: str = ".outputs/output_videos/"+DATASET_NAME+"_hdvideo"+str(VIDEO1_NUM).zfill(2)+str(VIDEO2_NUM).zfill(2)+"/"
OUTPUT_LANDMARKS_PATH: str = "sys_outputs/output_landmarks/"+DATASET_NAME+"_cam"+str(VIDEO1_NUM).zfill(2)+str(VIDEO2_NUM).zfill(2)+"/"

_CAMERA_PARAM_CACHE = {}
_LAZY_CAMERA_ATTRS = {
    "parameters",
    "param1",
    "param2",
    "K_left",
    "R_left",
    "T_left",
    "K_right",
    "R_right",
    "T_right",
}
_LAZY_MEDIAPIPE_ATTRS = {
    "mp_holistic",
    "mp_drawing",
    "mp_pose",
    "POSE_CONNECTIONS",
    "JOINT_STYLE",
    "BONE_STYLE",
}


def _Get_mediapipe():
    import mediapipe as mp

    return mp


def _Get_holistic():
    return _Get_mediapipe().solutions.holistic


def _Get_drawing():
    mp_drawing = _Get_mediapipe().solutions.drawing_utils
    mp_drawing._VISIBILITY_THRESHOLD = 0.0
    return mp_drawing


def _Get_pose():
    return _Get_mediapipe().solutions.pose


def _Get_pyplot():
    import matplotlib.pyplot as plt

    return plt


def Build_panoptic_paths(
    dataset_name: str = DEFAULT_DATASET_NAME,
    video1_num: int = DEFAULT_VIDEO1_NUM,
    video2_num: int = DEFAULT_VIDEO2_NUM,
    panoptic_root: str = DEFAULT_PANOPTIC_ROOT,
    inputs_root: str = DEFAULT_INPUTS_ROOT,
    videos_root: str = DEFAULT_VIDEOS_ROOT,
    landmarks_root: str = DEFAULT_LANDMARKS_ROOT,
) -> PanopticPathDict:
    """
    Panoptic Dataset のディレクトリ構造から、このモジュールで使うパスをまとめて生成する。

    Args:
        dataset_name: 使用する Panoptic Dataset のシーケンス名。例: `"171204_pose3"`。
        video1_num: 左カメラとして扱う HD カメラ番号。`18` なら `hd_00_18.mp4`。
        video2_num: 右カメラとして扱う HD カメラ番号。`23` なら `hd_00_23.mp4`。
        panoptic_root: Panoptic Dataset のルートディレクトリ。
        inputs_root: 抽出した入力画像を保存するルートディレクトリ。
        videos_root: 比較用動画を保存するルートディレクトリ。
        landmarks_root: 推定ランドマーク JSON を保存するルートディレクトリ。

    Returns:
        `video1_path`, `video2_path`, `calibration_path`,
        `input1_image_path`, `output_landmarks_path` などを含むパス辞書。

    Notes:
        ファイルの存在確認や読み込みは行わない。import 時の副作用を避けるため、
        実際に必要になったタイミングで呼び出す。
    """
    dataset_root = os.path.join(panoptic_root, dataset_name)
    video1_suffix = str(video1_num).zfill(2)
    video2_suffix = str(video2_num).zfill(2)

    return {
        "dataset_root": dataset_root,
        "video1_path": os.path.join(dataset_root, "hdVideos", f"hd_00_{video1_suffix}.mp4"),
        "video2_path": os.path.join(dataset_root, "hdVideos", f"hd_00_{video2_suffix}.mp4"),
        "calibration_path": os.path.join(dataset_root, f"calibration_{dataset_name}.json"),
        "gt_body_folder": os.path.join(dataset_root, "hdPose3d_stage1_coco19"),
        "gt_hand_folder": os.path.join(dataset_root, "hdHand3d"),
        "input1_image_path": os.path.join(inputs_root, f"{dataset_name}_cam{video1_suffix}"),
        "input2_image_path": os.path.join(inputs_root, f"{dataset_name}_cam{video2_suffix}"),
        "video_storage_path": os.path.join(videos_root, f"{dataset_name}_hdvideo{video1_suffix}{video2_suffix}"),
        "output_landmarks_path": os.path.join(landmarks_root, f"{dataset_name}_cam{video1_suffix}{video2_suffix}"),
    }


def Load_calibration_parameters(calibration_path: str) -> dict[str, Any]:
    """
    Panoptic Dataset のキャリブレーション JSON を読み込む。

    Args:
        calibration_path: `calibration_<dataset_name>.json` へのパス。

    Returns:
        JSON を `dict` として読み込んだもの。`"cameras"` キーを含む想定。

    Raises:
        FileNotFoundError: `calibration_path` が存在しない場合。
    """
    if not os.path.exists(calibration_path):
        raise FileNotFoundError(f"キャリブレーションファイルが見つかりません: {calibration_path}")

    with open(calibration_path) as calibration_file:
        return json.load(calibration_file)


def Load_camera_parameters(
    dataset_name: str = DEFAULT_DATASET_NAME,
    video1_num: int = DEFAULT_VIDEO1_NUM,
    video2_num: int = DEFAULT_VIDEO2_NUM,
    panoptic_root: str = DEFAULT_PANOPTIC_ROOT,
    calibration_path: Optional[str] = None,
) -> CameraParameterDict:
    """
    左右カメラの内部・外部パラメータを読み込む。

    Args:
        dataset_name: 使用する Panoptic Dataset のシーケンス名。
        video1_num: 左カメラとして扱う HD カメラ番号。
        video2_num: 右カメラとして扱う HD カメラ番号。
        panoptic_root: Panoptic Dataset のルートディレクトリ。
        calibration_path: キャリブレーション JSON を直接指定する場合のパス。
            `None` の場合は `dataset_name` と `panoptic_root` から生成する。

    Returns:
        次のキーを含む辞書。
        `parameters`: 読み込んだキャリブレーション JSON 全体。
        `param1`, `param2`: 左右カメラの元 JSON パラメータ。
        `K_left`, `K_right`: shape `(3, 3)` の内部パラメータ行列。
        `R_left`, `R_right`: shape `(3, 3)` の回転行列。
        `T_left`, `T_right`: shape `(3, 1)` 相当の並進ベクトル。

    Notes:
        同じ `calibration_path`, `video1_num`, `video2_num` の組み合わせは
        メモリ上にキャッシュする。
    """
    if calibration_path is None:
        paths = Build_panoptic_paths(
            dataset_name=dataset_name,
            video1_num=video1_num,
            video2_num=video2_num,
            panoptic_root=panoptic_root,
        )
        calibration_path = paths["calibration_path"]

    cache_key = (calibration_path, video1_num, video2_num)
    if cache_key in _CAMERA_PARAM_CACHE:
        return _CAMERA_PARAM_CACHE[cache_key]

    loaded_parameters = Load_calibration_parameters(calibration_path)
    left_param = loaded_parameters["cameras"][479 + video1_num]
    right_param = loaded_parameters["cameras"][479 + video2_num]
    camera_params = {
        "parameters": loaded_parameters,
        "param1": left_param,
        "param2": right_param,
        "K_left": np.asarray(left_param["K"]),
        "R_left": np.asarray(left_param["R"]),
        "T_left": np.asarray(left_param["t"]),
        "K_right": np.asarray(right_param["K"]),
        "R_right": np.asarray(right_param["R"]),
        "T_right": np.asarray(right_param["t"]),
    }
    _CAMERA_PARAM_CACHE[cache_key] = camera_params
    return camera_params


def __getattr__(name: str) -> Any:
    """
    既存コードの `hpe.K_left` などを壊しにくくするための互換レイヤー。
    import 時には読み込まず、属性参照時だけデフォルト設定のカメラパラメータをロードする。
    """
    if name in _LAZY_CAMERA_ATTRS:
        return Load_camera_parameters()[name]
    if name == "mp_holistic":
        return _Get_holistic()
    if name == "mp_drawing":
        return _Get_drawing()
    if name == "mp_pose":
        return _Get_pose()
    if name == "POSE_CONNECTIONS":
        return _Get_pose().POSE_CONNECTIONS
    if name == "JOINT_STYLE":
        return _Get_drawing().DrawingSpec(color=(255,0,0), thickness=5, circle_radius=3)
    if name == "BONE_STYLE":
        return _Get_drawing().DrawingSpec(color=(200,200,0), thickness=5)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def Estimate_fundamental_matrix_raw(
    points_left: Landmark2DArray,
    points_right: Landmark2DArray,
) -> np.ndarray:
    """
    非正規化・rank補正なしの8点アルゴリズムで基礎行列 `F` を推定する。

    Args:
        points_left: 左画像の対応点。shape `(N, 2)`、`N >= 8`。
        points_right: 右画像の対応点。shape `(N, 2)`、`N >= 8`。

    Returns:
        shape `(3, 3)` の未補正基礎行列 `F`。

    Notes:
        実装本体は `eight_point_pose.Estimate_fundamental_matrix_raw`。
        2D点の正規化と `rank(F) = 2` 補正は行わない。
    """
    from eight_point_pose import Estimate_fundamental_matrix_raw as _estimate

    return _estimate(points_left, points_right)


def Estimate_essential_matrix_raw(
    points_left: Landmark2DArray,
    points_right: Landmark2DArray,
    K_left: np.ndarray,
    K_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    正規化画像座標から未補正の本質行列 `E` を直接求める。

    Args:
        points_left: 左画像の対応点。shape `(N, 2)`。
        points_right: 右画像の対応点。shape `(N, 2)`。
        K_left: 左カメラの内部パラメータ。shape `(3, 3)`。
        K_right: 右カメラの内部パラメータ。shape `(3, 3)`。

    Returns:
        `(E, points_normalized_left, points_normalized_right)`。
        `points_*_normalized` は `inv(K) @ [u, v, 1]` で得た正規化画像座標。
        `E` は `A e = 0` を SVD で解いた、特異値補正なしの本質行列。
    """
    from eight_point_pose import Estimate_essential_matrix_raw as _estimate

    return _estimate(points_left, points_right, K_left, K_right)


def Decompose_essential_matrix_raw(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    特異値補正なしの本質行列から4通りの `(R, t)` 候補を作る。

    Args:
        E: shape `(3, 3)` の本質行列。

    Returns:
        `(R, t)` 候補のリスト。`R` は shape `(3, 3)`、`t` は shape `(3,)`。
    """
    from eight_point_pose import Decompose_essential_matrix_raw as _decompose

    return _decompose(E)


def Estimate_relative_pose_eight_point_raw(
    points_left: Landmark2DArray,
    points_right: Landmark2DArray,
    K_left: np.ndarray,
    K_right: np.ndarray,
) -> dict[str, Any]:
    """
    非正規化・補正なし8点アルゴリズムで左右カメラの相対姿勢を推定する。

    Args:
        points_left: 左画像の対応点。shape `(N, 2)`、`N >= 8`。
        points_right: 右画像の対応点。shape `(N, 2)`、`N >= 8`。
        K_left: 左カメラの内部パラメータ。shape `(3, 3)`。
        K_right: 右カメラの内部パラメータ。shape `(3, 3)`。

    Returns:
        `R`, `t`, `E`, `points_normalized_left`, `points_normalized_right`,
        `points3d`, `positive_depth_count` などを含む辞書。
        比較用に `E_rank`, `E_singular_values`, `det_R`,
        `rotation_orthogonality_error` も含む。

    Notes:
        Hartley 正規化と `E` の特異値補正は行わない。
        ただし、既知の内部パラメータ `K` による正規化画像座標への変換は行う。
    """
    from eight_point_pose import Estimate_relative_pose_eight_point_raw as _estimate

    return _estimate(points_left, points_right, K_left, K_right)


def Estimate_external_parameters_eight_point_raw(
    points_left: Landmark2DArray,
    points_right: Landmark2DArray,
    K_left: np.ndarray,
    K_right: np.ndarray,
) -> dict[str, Any]:
    """
    `Estimate_relative_pose_eight_point_raw` の用途名を明確にした別名。

    Returns:
        外部パラメータ相当の相対姿勢 `R`, `t` と診断情報を含む辞書。
    """
    return Estimate_relative_pose_eight_point_raw(points_left, points_right, K_left, K_right)


# ボーン情報
#: Panoptic body GT を描画するときのボーン接続。shape `(14, 2)`。
GT_POSE_CONNECTIONS: np.ndarray = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1
#: Panoptic hand GT を描画するときのボーン接続。shape `(20, 2)`。
GT_HAND_CONNECTIONS: np.ndarray = np.array([[1,2],[2,3],[3,4],[4,5],
                                [1,6],[6,7],[7,8],[8,9],
                                [1,10],[10,11],[11,12],[12,13],
                                [1,14],[14,15],[15,16],[16,17],
                                [1,18],[18,19],[19,20],[20,21]])-1

# 処理関数
def Landmark_detect(image_left: ImageArray, image_right: ImageArray) -> tuple[Any, Any]:
    """
    左右画像から MediaPipe Holistic のランドマークを検出する。

    Args:
        image_left: 左カメラ画像。OpenCV で読み込んだ BGR 配列、shape は `(H, W, 3)`。
        image_right: 右カメラ画像。OpenCV で読み込んだ BGR 配列、shape は `(H, W, 3)`。

    Returns:
        `(result_left, result_right)` のタプル。
        各要素は MediaPipe Holistic の結果オブジェクトで、
        `pose_landmarks`, `left_hand_landmarks`, `right_hand_landmarks`,
        `face_landmarks` などを参照できる。

    Raises:
        ValueError: 左右どちらかの画像が `None` の場合。

    Notes:
        入力は BGR を想定し、関数内で RGB に変換してから MediaPipe に渡す。
    """
    if image_left is None or image_right is None:
        raise ValueError("入力画像が None です。左右両方の BGR 画像を指定してください。")

    # RGBスケールに変換
    image_leftRGB = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)
    image_rightRGB = cv2.cvtColor(image_right, cv2.COLOR_BGR2RGB)

    # ランドマーク推定 静止画入力 -> static_image_mode=True
    mp_holistic = _Get_holistic()
    with mp_holistic.Holistic(static_image_mode=True) as holistic:
        result_left = holistic.process(image_leftRGB)
        result_right = holistic.process(image_rightRGB)

    return result_left, result_right

def Normalized_to_screen_coord(
    result_L: Any,
    result_R: Any,
    width: int,
    height: int,
) -> tuple[Landmark2DArray, Landmark2DArray]:
    """
    MediaPipe の正規化ランドマークを画像座標系の `(x, y)` に変換する。

    Args:
        result_L: 左画像のランドマークリスト。`.landmark` を持つ MediaPipe オブジェクト。
        result_R: 右画像のランドマークリスト。`.landmark` を持つ MediaPipe オブジェクト。
        width: 入力画像の幅。正規化 x 座標に掛ける。
        height: 入力画像の高さ。正規化 y 座標に掛ける。

    Returns:
        `(scr_landmarks_L, scr_landmarks_R)`。
        どちらも shape `(N, 2)` の `numpy.ndarray` で、各行は `[x, y]`。

    Notes:
        `pose_landmarks` を渡した場合、通常 `N=33` になる。
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

def Matching_landmarks(
    landmark_left: Any,
    landmark_right: Any,
    enable: bool,
    verbose: bool = False,
) -> list[int]:
    """
    左右ランドマークの visibility を使って、有効な対応点を示すリストを作る。

    Args:
        landmark_left: 左画像の MediaPipe ランドマーク。各点に `.visibility` がある想定。
        landmark_right: 右画像の MediaPipe ランドマーク。各点に `.visibility` がある想定。
        enable: `True` の場合、左右どちらも visibility が閾値を超える点だけ `1` にする。
            `False` の場合、すべての点を `1` として扱う。
        verbose: `True` の場合、各ランドマークの visibility を標準出力に表示する。

    Returns:
        各ランドマークが有効なら `1`、無効なら `0` のリスト。

    Notes:
        現在の閾値は `0.80`。
    """
    # 閾値
    visibility_threshold = 0.80  # 80%以上
    matching_list = []
    if enable:
        for l, r in zip(landmark_left.landmark, landmark_right.landmark):
            if verbose:
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

def Projection_mat_calc(
    K_left: np.ndarray,
    R_left: np.ndarray,
    T_left: np.ndarray,
    K_right: np.ndarray,
    R_right: np.ndarray,
    T_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    左右カメラの投影行列 `P = K [R|t]` を生成する。

    Args:
        K_left: 左カメラの内部パラメータ行列。shape `(3, 3)`。
        R_left: 左カメラの回転行列。shape `(3, 3)`。
        T_left: 左カメラの並進ベクトル。shape `(3, 1)` または `(3,)` 相当。
        K_right: 右カメラの内部パラメータ行列。shape `(3, 3)`。
        R_right: 右カメラの回転行列。shape `(3, 3)`。
        T_right: 右カメラの並進ベクトル。shape `(3, 1)` または `(3,)` 相当。

    Returns:
        `(Pleft, Pright)`。
        どちらも shape `(3, 4)` の投影行列。
    """
    
    
    # 左カメラをワールド座標とするため、左投影行列 P_left = K[I|0] = [K|0]　となる
    Pleft = np.hstack((K_left @ R_left, K_left @ T_left))
    Pright = np.hstack((K_right @ R_right, K_right @ T_right))

    return Pleft, Pright

def Triangulate_3Dpoint(
    Pleft: np.ndarray,
    Pright: np.ndarray,
    landmark_left: Landmark2DArray,
    landmark_right: Landmark2DArray,
) -> Landmark3DArray:
    """
    左右画像の対応する2D点から3D点を三角測量する。

    Args:
        Pleft: 左カメラの投影行列。shape `(3, 4)`。
        Pright: 右カメラの投影行列。shape `(3, 4)`。
        landmark_left: 左画像の2Dランドマーク。shape `(N, 2)`、各行は `[x, y]`。
        landmark_right: 右画像の2Dランドマーク。shape `(N, 2)`、各行は `[x, y]`。

    Returns:
        三角測量された3Dランドマーク。shape `(N, 3)`。

    Raises:
        ValueError: 左右ランドマークの shape が違う場合、または `(N, 2)` ではない場合。

    Notes:
        `cv2.triangulatePoints` を使うため、左右の点は同じ順序で対応している必要がある。
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
    
    landmark_left = np.asarray(landmark_left, dtype=float)
    landmark_right = np.asarray(landmark_right, dtype=float)
    if landmark_left.shape != landmark_right.shape:
        raise ValueError(f"左右ランドマークの形状が一致しません: {landmark_left.shape} != {landmark_right.shape}")
    if landmark_left.ndim != 2 or landmark_left.shape[1] != 2:
        raise ValueError(f"ランドマークは (N, 2) の配列で指定してください: {landmark_left.shape}")

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

def Triangulate3DHPE(
    img1: ImageArray,
    img2: ImageArray,
    K_l: np.ndarray,
    R_l: np.ndarray,
    T_l: np.ndarray,
    K_r: np.ndarray,
    R_r: np.ndarray,
    T_r: np.ndarray,
    verbose: bool = False,
) -> tuple[Any, Any, Landmark3DArray]:
    """
    左右画像から MediaPipe Pose を検出し、全身33点の3D座標を推定する。

    Args:
        img1: 左カメラ画像。OpenCV BGR 配列、shape `(H, W, 3)`。
        img2: 右カメラ画像。OpenCV BGR 配列、shape `(H, W, 3)`。
        K_l: 左カメラの内部パラメータ。shape `(3, 3)`。
        R_l: 左カメラの回転行列。shape `(3, 3)`。
        T_l: 左カメラの並進ベクトル。shape `(3, 1)` または `(3,)` 相当。
        K_r: 右カメラの内部パラメータ。shape `(3, 3)`。
        R_r: 右カメラの回転行列。shape `(3, 3)`。
        T_r: 右カメラの並進ベクトル。shape `(3, 1)` または `(3,)` 相当。
        verbose: `True` の場合、2D/3Dランドマークなどのデバッグ情報を表示する。

    Returns:
        `(result_left, result_right, landmark3D)`。
        `result_left` と `result_right` は MediaPipe Holistic の結果オブジェクト。
        `landmark3D` は shape `(33, 3)` の `numpy.ndarray`。
        人物を検出できない場合は shape `(0, 3)` の空配列。
    """
    # 高さ、幅（同じカメラを用いるため片方の画像から取得）
    HEIGHT, WIDTH, _ = img1.shape

    # ランドマーク、マッチングリスト
    result_left, result_right = Landmark_detect(image_left=img1, image_right=img2)

    if result_left.pose_landmarks is None or result_right.pose_landmarks is None:
        if verbose:
            print("Enter no human!")
        landmark3D = np.empty((0, 3))
    else:
        # 画像座標系への変換(同時に体のランドマークのみ使用)
        pose_left, pose_right = Normalized_to_screen_coord(result_left.pose_landmarks, result_right.pose_landmarks, WIDTH, HEIGHT)
        if verbose:
            print("left=","\n",pose_left)
            print("right=","\n",pose_right)

        # 投影行列を計算
        Pleft, Pright = Projection_mat_calc(K_l, R_l, T_l, K_r, R_r, T_r)

        # 3次元位置を復元
        landmark3D = Triangulate_3Dpoint(Pleft, Pright, pose_left, pose_right)
        if verbose:
            print("landmark3D:","\n",landmark3D)

    return result_left, result_right, landmark3D
# 結果表示用関数
def annotate_image(
    image_L: ImageArray,
    landmarks_L: Any,
    image_R: ImageArray,
    landmarks_R: Any,
    connections: Any,
    jointstyle: Any,
    bonestyle: Any,
    temp_storage_path: str = TEMPORARY_IMAGES_STORAGE_PATH,
) -> tuple[ImageArray, ImageArray]:
    """
    MediaPipe のランドマークを左右画像へ描画し、注釈済み画像を返す。

    Args:
        image_L: 左カメラ画像。OpenCV BGR 配列。
        landmarks_L: 左画像の MediaPipe ランドマーク。例: `result_left.pose_landmarks`。
        image_R: 右カメラ画像。OpenCV BGR 配列。
        landmarks_R: 右画像の MediaPipe ランドマーク。例: `result_right.pose_landmarks`。
        connections: 描画するボーン接続情報。例: `POSE_CONNECTIONS`。
        jointstyle: ランドマーク点の描画スタイル。例: `JOINT_STYLE`。
        bonestyle: ボーン線の描画スタイル。例: `BONE_STYLE`。
        temp_storage_path: 元画像・注釈画像を一時保存するディレクトリ。

    Returns:
        `(image_L_annotation, image_R_annotation)`。
        どちらもランドマーク描画後の OpenCV BGR 画像。

    Notes:
        互換性維持のため一時 PNG を保存してから読み戻す実装になっている。
    """
    os.makedirs(temp_storage_path, exist_ok=True)
    original_left_path = os.path.join(temp_storage_path, "original_L.png")
    original_right_path = os.path.join(temp_storage_path, "original_R.png")
    annotated_left_path = os.path.join(temp_storage_path, "annotated_L.png")
    annotated_right_path = os.path.join(temp_storage_path, "annotated_R.png")

    cv2.imwrite(original_left_path, image_L)
    cv2.imwrite(original_right_path, image_R)
    # cv2.imshow("L original_image", original_images[0])
    mp_drawing = _Get_drawing()
    mp_drawing.draw_landmarks(image_L, landmarks_L, connections, jointstyle, bonestyle)
    mp_drawing.draw_landmarks(image_R, landmarks_R, connections, jointstyle, bonestyle)  
    cv2.imwrite(annotated_left_path, image_L)
    cv2.imwrite(annotated_right_path, image_R)
    # cv2.imshow("L annotated_image", annotated_images[0])
    image_L_annotation = cv2.imread(annotated_left_path)
    image_R_annotation = cv2.imread(annotated_right_path)
    return image_L_annotation, image_R_annotation

def plot_2Dskeleton(landmarks: Landmark2DArray, connection: Sequence[Sequence[int]]) -> None:
    """
    2Dランドマークとボーン接続を Matplotlib で表示する。

    Args:
        landmarks: 画像座標系の2Dランドマーク。shape `(N, 2)`。
        connection: 接続するランドマーク番号のペア。例: `[[0, 1], [1, 2]]`。

    Returns:
        なし。Matplotlib のウィンドウを表示する。
    """
    plt = _Get_pyplot()
    x_coords = []
    y_coords = []
    fig = plt.figure()
    ax = fig.add_subplot()
    # ランドマークをプロット
    for i, landmark in enumerate(landmarks):
        x_coords.append(landmark[0])
        y_coords.append(landmark[1])

        fig.text(landmark[0], landmark[1], str(i), color="black", fontsize=8, ha='right', va='bottom')
        
    # キーポイントを描画
    ax.scatter(x_coords, y_coords, color='red')

    # ボーンを描画
    for start_idx, end_idx in connection:
        ax.plot([x_coords[start_idx], x_coords[end_idx]],
                    [y_coords[start_idx], y_coords[end_idx]],
                    color='blue')
        
    ax.invert_yaxis()  # Y座標を反転して画像と一致させる
    ax.axis('equal')         # アスペクト比を正確にする
    # fig.savefig()
    plt.show()
    cv2.destroyAllWindows()

def plot_3Dskeleton(
    frame_num: int,
    output_landmarks: Landmark3DArray,
    gt_landmarks: np.ndarray,
    gt_left: np.ndarray,
    gt_right: np.ndarray,
    output_connections: Sequence[Sequence[int]],
    gt_connections: Sequence[Sequence[int]],
    gt_hand_connections: Sequence[Sequence[int]],
) -> tuple[Any, Any]:
    """
    推定3Dランドマークと GT ランドマークを同じ3Dプロットに描画する。

    Args:
        frame_num: 表示対象のフレーム番号。
        output_landmarks: 推定された MediaPipe 3Dランドマーク。shape `(33, 3)`。
        gt_landmarks: GT の body ランドマーク。shape `(19, 3)` または `(19, 4)`。
        gt_left: GT の左手ランドマーク。shape `(21, 3)`。
        gt_right: GT の右手ランドマーク。shape `(21, 3)`。
        output_connections: 推定ランドマーク用の接続ペア。
        gt_connections: GT body 用の接続ペア。
        gt_hand_connections: GT hand 用の接続ペア。

    Returns:
        `(fig, ax)`。Matplotlib の Figure と 3D Axes。
    """
    plt = _Get_pyplot()
    fig = plt.figure(figsize = (8, 8))
    ax= fig.add_subplot(111, projection='3d')
    
    # 推定結果をプロットに表示
    ax.scatter(output_landmarks[:, 0], output_landmarks[:,1],output_landmarks[:,2], s = 10, c = "red")
    for connection in output_connections:
        start_joint, end_joint = connection
        x = [output_landmarks[start_joint, 0], output_landmarks[end_joint, 0]]
        y = [output_landmarks[start_joint, 1], output_landmarks[end_joint, 1]]
        z = [output_landmarks[start_joint, 2], output_landmarks[end_joint, 2]]
        ax.plot(x, y, z, c='red', linewidth=2)

    # グラウンドトゥルースをプロットに表示
    ax.scatter(gt_landmarks[:, 0], gt_landmarks[:,1],gt_landmarks[:,2], s = 10, c = "blue") # ボディ
    for connection in gt_connections:
        start_joint, end_joint = connection
        x = [gt_landmarks[start_joint, 0], gt_landmarks[end_joint, 0]]
        y = [gt_landmarks[start_joint, 1], gt_landmarks[end_joint, 1]]
        z = [gt_landmarks[start_joint, 2], gt_landmarks[end_joint, 2]]
        ax.plot(x, y, z, c='blue', linewidth=2)
    ax.scatter(gt_left[:, 0], gt_left[:,1],gt_left[:,2], s = 10, c = "blue") # 左手
    for connection in gt_hand_connections:
        start_joint, end_joint = connection
        x = [gt_left[start_joint, 0], gt_left[end_joint, 0]]
        y = [gt_left[start_joint, 1], gt_left[end_joint, 1]]
        z = [gt_left[start_joint, 2], gt_left[end_joint, 2]]
        ax.plot(x, y, z, c='blue', linewidth=2)
    ax.scatter(gt_right[:, 0], gt_right[:,1],gt_right[:,2], s = 10, c = "blue") # 右手
    for connection in gt_hand_connections:
        start_joint, end_joint = connection
        x = [gt_right[start_joint, 0], gt_right[end_joint, 0]]
        y = [gt_right[start_joint, 1], gt_right[end_joint, 1]]
        z = [gt_right[start_joint, 2], gt_right[end_joint, 2]]
        ax.plot(x, y, z, c='blue', linewidth=2)

        # if not (gt_right[start_joint] == [0,0,0] or gt_right[end_joint] == [0,0,0]): # [0,0,0]をプロットから省く
    set_equal_aspect(ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("frame"+str(frame_num).zfill(8))
    # 対象を正面から見た3Dプロットを画像として保存
    ax.view_init(elev=180, azim=5, roll=-90)

    return fig, ax

# プロットの軸を等間隔にする
def set_equal_aspect(ax: Any) -> None:
    """
    Matplotlib の3D軸で x, y, z の表示スケールを揃える。

    Args:
        ax: Matplotlib の 3D Axes。

    Returns:
        なし。`ax` を直接変更する。
    """
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

def concatenate_images(
    frame_num: int,
    input1_image_path: str = INPUT1_IMAGE_PATH,
    input2_image_path: str = INPUT2_IMAGE_PATH,
) -> ImageArray:
    """
    指定フレームの左右画像を読み込み、横方向に連結する。

    Args:
        frame_num: 読み込むフレーム番号。ファイル名は `frameXXXXXXXX.png`。
        input1_image_path: 左画像が保存されているディレクトリ。
        input2_image_path: 右画像が保存されているディレクトリ。

    Returns:
        `cv2.hconcat([image2, image1])` で連結した OpenCV BGR 画像。

    Raises:
        FileNotFoundError: 左右どちらかの画像を読み込めない場合。
    """
    # 入力フレームの読み込み
    image1 = cv2.imread(os.path.join(input1_image_path, "frame"+str(frame_num).zfill(8)+".png"))
    image2 = cv2.imread(os.path.join(input2_image_path, "frame"+str(frame_num).zfill(8)+".png"))

    if image1 is None or image2 is None:
        raise FileNotFoundError(f"連結する画像を読み込めませんでした: frame {frame_num}")

    # 画像を横に連結
    concatenation_image = cv2.hconcat([image2, image1])

    return concatenation_image

def create_video_from_images(
    concatenation_left: Sequence[ImageArray],
    concatenation_right: Sequence[ImageArray],
    frame_rate: int,
    save_path: str,
) -> None:
    """
    画像列から左右比較用の MP4 動画を作成する。

    Args:
        concatenation_left: 左側出力動画に書き込む画像列。
        concatenation_right: 右側出力動画に書き込む画像列。
        frame_rate: 出力動画の fps。
        save_path: `output_left_video.mp4` と `output_right_video.mp4` を保存するディレクトリ。

    Returns:
        なし。動画ファイルを `save_path` に保存する。

    Raises:
        ValueError: 書き込む画像がない場合。
    """
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

    if len(frames_left) == 0 or len(frames_right) == 0:
        raise ValueError("動画化する画像がありません。")

    os.makedirs(save_path, exist_ok=True)
    HEIGHT, WIDTH, _ = frames_left[0].shape
    out_Lvideo = cv2.VideoWriter(os.path.join(save_path, "output_left_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (WIDTH, HEIGHT))
    out_Rvideo = cv2.VideoWriter(os.path.join(save_path, "output_right_video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (WIDTH, HEIGHT))

    try:
        for i in range(len(frames_left)):
            out_Lvideo.write(frames_left[i])
            out_Rvideo.write(frames_right[i])
    finally:
        out_Lvideo.release()
        out_Rvideo.release()
# create_video_from_images(storage_path=FRAMES_STORAGE_PATH)

# 集計関数
def compute_mpjpe(predicted: Landmark3DArray, ground_truth: Landmark3DArray) -> float:
    """
    MPJPE (Mean Per Joint Position Error) を計算する。

    Args:
        predicted: 予測された3D関節位置。shape `(N, 3)`。
        ground_truth: GT の3D関節位置。shape `(N, 3)`。

    Returns:
        各関節のユークリッド距離の平均値。座標が mm なら戻り値も mm。

    Raises:
        AssertionError: `predicted` と `ground_truth` の shape が一致しない場合。
    """
    assert predicted.shape == ground_truth.shape, "形状が一致しません"
    errors = np.linalg.norm(predicted - ground_truth, axis=1)
    mpjpe = np.mean(errors)
    return mpjpe

def make_compe_landmarks(
    ground_truth: np.ndarray,
    predicted: Landmark3DArray,
    compe_map: Sequence[Sequence[int]],
    confidence: float = -1.0,
) -> tuple[np.ndarray, Landmark3DArray]:
    """
    GT と推定値から、同じ身体部位に対応するランドマークだけを抽出する。

    Args:
        ground_truth: GT ランドマーク。shape `(N, 4)` を想定し、4列目を信頼度として使う。
        predicted: 推定3Dランドマーク。shape `(M, 3)`。
        compe_map: `[gt_index, predicted_index]` の対応表。
        confidence: GT の信頼度閾値。この値以下の GT ランドマークは除外する。

    Returns:
        `(compe_ground_truth, compe_predicted)`。
        `compe_ground_truth` は抽出後の GT 配列、`compe_predicted` は抽出後の推定配列。
    """

    compe_predicted = []
    compe_ground_truth = []


    for [gt_num, pd_num] in compe_map:
        # === GT信頼度フィルタ ===
        if ground_truth [gt_num, 3] <= confidence:
            # print("Throught!!")
            continue

        compe_predicted.append(predicted[pd_num])
        compe_ground_truth.append(ground_truth[gt_num])

    return np.array(compe_ground_truth), np.array(compe_predicted)

def read_gt_landmarks(
    frame_num: int,
    gt_body_folder: str = GT_DATA_FOLDER_PATH,
    gt_hand_folder: str = GT_HAND_DATA_FOLDER_PATH,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    指定フレームの Panoptic GT body/hand ランドマークを読み込む。

    Args:
        frame_num: 読み込むフレーム番号。
        gt_body_folder: `body3DScene_XXXXXXXX.json` があるディレクトリ。
        gt_hand_folder: `handRecon3D_hdXXXXXXXX.json` があるディレクトリ。
        verbose: `True` の場合、JSON 読み込み結果を標準出力に表示する。

    Returns:
        `(gt_body, gt_left, gt_right)`。
        `gt_body` は shape `(19, 4)` の body ランドマーク。
        `gt_left`, `gt_right` は shape `(21, 3)` の手ランドマーク。
        データがない場合はゼロ配列を返す。
    """
    # 正解データ呼び出し
    body_path = os.path.join(gt_body_folder, "body3DScene_"+str(frame_num).zfill(8)+".json")
    hand_path = os.path.join(gt_hand_folder, "handRecon3D_hd"+str(frame_num).zfill(8)+".json")

    with open(body_path) as gt: # ボディ
        ground_truth_file = gt
        gt_frame = json.load(ground_truth_file)
        if verbose:
            print(not(gt_frame is None))
    # (x,y,z,c)の19行4列の配列を作成
    if len(gt_frame['bodies']) == 0:
        gt_body = np.zeros((19, 4))
    else:
        gt_body = np.array(gt_frame['bodies'][0]['joints19']).reshape(-1, 4)
    with open(hand_path) as gt: # 両手
        ground_truth_file = gt
        gt_frame = json.load(ground_truth_file)
        if verbose:
            print(not(gt_frame is None))
    # (x,y,z)の21行3列の配列を作成 'landmarks'の長さが63であり、手のランドマークは21個であることから想定
    if len(gt_frame['people']) == 0:
        gt_left = np.zeros((21, 3))
        gt_right = np.zeros((21, 3))
    else:
        if not(len(gt_frame['people'][0]['left_hand']) == 0):
            gt_left = np.array(gt_frame['people'][0]['left_hand']['landmarks']).reshape(-1, 3)
        else:
            gt_left = np.zeros((21,3))
        if not(len(gt_frame['people'][0]['right_hand']) == 0):
            gt_right = np.array(gt_frame['people'][0]['right_hand']['landmarks']).reshape(-1, 3)
        else:
            gt_right = np.zeros((21,3))
    
    return gt_body, gt_left, gt_right

def read_pd_landmarks(
    frame_num: int,
    prediction_folder: str = OUTPUT_LANDMARKS_PATH,
    verbose: bool = False,
) -> Landmark3DArray:
    """
    指定フレームの推定ランドマーク JSON を読み込む。

    Args:
        frame_num: 読み込むフレーム番号。
        prediction_folder: `frame_XXXXXXXX.json` があるディレクトリ。
        verbose: `True` の場合、JSON 読み込み結果を標準出力に表示する。

    Returns:
        shape `(33, 3)` の推定3Dランドマーク。
        JSON の `"landmarks"` が `null` の場合はゼロ配列を返す。
    """
    # 推定ランドマーク呼び出し
    pred_path = os.path.join(prediction_folder, "frame_"+str(frame_num).zfill(8)+".json")
    with open(pred_path) as out:
        out_frame_data = json.load(out)
        if verbose:
            print(not(out_frame_data is None))
    # (x,y,z)の33行3列の配列を作成
    if out_frame_data['landmarks'] == None:
        out_body = np.zeros((33, 3))
    else:
        out_body_list = []
        for lm_num, lm in out_frame_data['landmarks'].items():
            out_body_list.append(lm)
        out_body = np.array(out_body_list).reshape(-1, 3)
    return out_body

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
    test_mode: bool = False,
    verbose: bool = False,
) -> List[int]:
    """
    Panoptic 動画からフレームを読み込み、3Dランドマーク推定結果を JSON に保存する。

    Args:
        dataset_name: 使用する Panoptic Dataset のシーケンス名。
        video1_num: 左カメラとして扱う HD カメラ番号。
        video2_num: 右カメラとして扱う HD カメラ番号。
        capture_rate: 何フレームごとに処理するか。例: `30` なら30フレーム間隔。
        first_frame_num: 処理開始フレーム番号。
        end_frame_num: 処理終了フレーム番号。この値は処理対象に含まない。
        panoptic_root: Panoptic Dataset のルートディレクトリ。
        inputs_root: 抽出した入力画像を保存するルートディレクトリ。
        videos_root: 比較用動画を保存するルートディレクトリ。
        landmarks_root: 推定ランドマーク JSON を保存するルートディレクトリ。
        temp_storage_path: 中間画像を保存するディレクトリ。
        test_mode: `True` の場合、`./testIO` 配下のテスト用ディレクトリ構造を使う。
        verbose: `True` の場合、処理中のフレーム番号などを標準出力に表示する。

    Returns:
        推定結果 JSON を保存したフレーム番号のリスト。

    Output JSON:
        各フレームを `frame_XXXXXXXX.json` として保存する。
        形式は `{"frame": frame_num, "landmarks": {"0": [x, y, z], ...}}`。
        人物未検出時は `"landmarks": null`。

    Raises:
        FileNotFoundError: 動画またはキャリブレーション JSON を開けない場合。
    """
    if test_mode:
        test_root = "./testIO"
        inputs_root = os.path.join(test_root, "test_sys_inputs")
        videos_root = os.path.join(test_root, "test_sys_outputs", "output_videos")
        landmarks_root = os.path.join(test_root, "test_sys_outputs", "output_landmarks")
        temp_storage_path = os.path.join(test_root, "test_sys_outputs", "images_temporary_storage")

    paths = Build_panoptic_paths(
        dataset_name=dataset_name,
        video1_num=video1_num,
        video2_num=video2_num,
        panoptic_root=panoptic_root,
        inputs_root=inputs_root,
        videos_root=videos_root,
        landmarks_root=landmarks_root,
    )

    # フォルダ生成
    os.makedirs(paths["video_storage_path"], exist_ok=True)
    os.makedirs(paths["input1_image_path"], exist_ok=True)
    os.makedirs(paths["input2_image_path"], exist_ok=True)
    os.makedirs(paths["output_landmarks_path"], exist_ok=True)
    os.makedirs(temp_storage_path, exist_ok=True)

    # キャリブレーションファイルをロード
    camera_params = Load_camera_parameters(
        dataset_name=dataset_name,
        video1_num=video1_num,
        video2_num=video2_num,
        panoptic_root=panoptic_root,
        calibration_path=paths["calibration_path"],
    )
    if verbose:
        print("cam_params:", camera_params["parameters"] is not None)

    video1 = cv2.VideoCapture(paths["video1_path"]) # カメラ1動画
    video2 = cv2.VideoCapture(paths["video2_path"]) # カメラ2動画

    if not video1.isOpened() or not video2.isOpened():
        video1.release()
        video2.release()
        raise FileNotFoundError(
            "指定した動画ファイルを開けませんでした。"
            f" video1={paths['video1_path']}, video2={paths['video2_path']}"
        )

    processed_frames: List[int] = []
    frame_num = first_frame_num

    try:
        while frame_num < end_frame_num:
            video1.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            video2.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
            if verbose:
                print("now frame:" + str(video1.get(cv2.CAP_PROP_POS_FRAMES)))

            ret1, img1 = video1.read()
            ret2, img2 = video2.read()
            if verbose:
                print("now frame:" + str(video1.get(cv2.CAP_PROP_POS_FRAMES)))

            if not (ret1 and ret2):
                if verbose:
                    print("breaked frame:", frame_num)
                break

            cv2.imwrite(os.path.join(paths["input1_image_path"], f"frame{frame_num:08d}.png"), img1)
            cv2.imwrite(os.path.join(paths["input2_image_path"], f"frame{frame_num:08d}.png"), img2)

            _, _, landmarks = Triangulate3DHPE(
                img1,
                img2,
                camera_params["K_left"],
                camera_params["R_left"],
                camera_params["T_left"],
                camera_params["K_right"],
                camera_params["R_right"],
                camera_params["T_right"],
                verbose=verbose,
            )

            landmark_dict = None
            if landmarks is not None and len(landmarks) != 0:
                landmark_dict = {str(idx): [float(landmark[0]), float(landmark[1]), float(landmark[2])]
                                 for idx, landmark in enumerate(landmarks)}

            result = {"frame": frame_num, "landmarks": landmark_dict}
            with open(os.path.join(paths["output_landmarks_path"], f"frame_{frame_num:08d}.json"), "w") as f:
                json.dump(result, f, indent=2)

            processed_frames.append(frame_num)
            frame_num += capture_rate
    finally:
        video1.release()
        video2.release()

    if verbose:
        print("finish!!")
    return processed_frames    

def run_mpjpe_evaluation(
    first_frame: int = DEFAULT_FIRST_FRAME_NUM,
    end_frame: int = DEFAULT_END_FRAME_NUM,
    capture_rate: int = DEFAULT_CAPTURE_RATE,
    gt_body_folder: str = GT_DATA_FOLDER_PATH,
    gt_hand_folder: str = GT_HAND_DATA_FOLDER_PATH,
    prediction_folder: str = OUTPUT_LANDMARKS_PATH,
    body_map: list[list[int]] = BODY_MAP,
    left_hand_map: list[list[int]] = LEFT_HAND_MAP,
    right_hand_map: list[list[int]] = RIGHT_HAND_MAP,
    confidence_threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """
    保存済み推定ランドマークと Panoptic GT から MPJPE をフレームごとに計算する。

    Args:
        first_frame: 評価開始フレーム番号。
        end_frame: 評価終了フレーム番号。この値は評価対象に含まない。
        capture_rate: 何フレームごとに評価するか。
        gt_body_folder: GT body JSON があるディレクトリ。
        gt_hand_folder: GT hand JSON があるディレクトリ。
        prediction_folder: 推定ランドマーク JSON があるディレクトリ。
        body_map: body の `[gt_index, predicted_index]` 対応表。
        left_hand_map: 左手の `[gt_index, predicted_index]` 対応表。
        right_hand_map: 右手の `[gt_index, predicted_index]` 対応表。
        confidence_threshold: GT ランドマークの信頼度閾値。

    Returns:
        各フレームの評価結果の辞書リスト。
        正常時は `body_mpjpe`, `left_hand_mpjpe`, `right_hand_mpjpe`,
        `holistic_mpjpe`, `status` などを含む。
        読み込み失敗時は `status` に `missing_gt_body`,
        `missing_gt_hand`, `missing_prediction` などが入る。
    """
    results: list[dict] = []
    frame_num = first_frame

    # # パス生成
    # dataset_root = os.path.join(panoptic_root, dataset_name)
    # video1_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video1_num).zfill(2)}.mp4")
    # video2_path = os.path.join(dataset_root, "hdVideos", f"hd_00_{str(video2_num).zfill(2)}.mp4")
    # calibration_path = os.path.join(dataset_root, f"calibration_{dataset_name}.json")

    # input1_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}")
    # input2_image_path = os.path.join(inputs_root, f"{dataset_name}_cam{str(video2_num).zfill(2)}")
    # video_storage_path = os.path.join(videos_root, f"{dataset_name}_hdvideo{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")
    # output_landmarks_path = os.path.join(landmarks_root, f"{dataset_name}_cam{str(video1_num).zfill(2)}{str(video2_num).zfill(2)}")

    while frame_num < end_frame:
        result_row: dict = {"frame": frame_num, "status": "ok"}
        mpjpe_flag = True

        body_path = os.path.join(gt_body_folder, f"body3DScene_{frame_num:08d}.json")
        hand_path = os.path.join(gt_hand_folder, f"handRecon3D_hd{frame_num:08d}.json")
        pred_path = os.path.join(prediction_folder, f"frame_{frame_num:08d}.json")

        try:
            with open(body_path) as body_file:
                gt_body_json = json.load(body_file)
        except FileNotFoundError:
            result_row["status"] = "missing_gt_body"
            results.append(result_row)
            frame_num += capture_rate
            continue

        if len(gt_body_json.get("bodies", [])) == 0:
            gt_body = np.zeros((19, 4))
            mpjpe_flag = False
        else:
            gt_body = np.asarray(gt_body_json["bodies"][0]["joints19"]).reshape(-1, 4)

        try:
            with open(hand_path) as hand_file:
                gt_hand_json = json.load(hand_file)
        except FileNotFoundError:
            result_row["status"] = "missing_gt_hand"
            results.append(result_row)
            frame_num += capture_rate
            continue

        if len(gt_hand_json.get("people", [])) == 0:
            gt_left = np.zeros((21, 4))
            gt_right = np.zeros((21, 4))
            mpjpe_flag = False
        else:
            person = gt_hand_json["people"][0]

            if person.get("left_hand"):
                left_landmarks = np.asarray(person["left_hand"]["landmarks"]).reshape(-1, 3)
                left_conf = np.asarray(person["left_hand"]["averageScore"]).reshape(-1, 1)
                gt_left = np.concatenate((left_landmarks, left_conf), axis=1)
            else:
                gt_left = np.zeros((21, 4))
                mpjpe_flag = False

            if person.get("right_hand"):
                right_landmarks = np.asarray(person["right_hand"]["landmarks"]).reshape(-1, 3)
                right_conf = np.asarray(person["right_hand"]["averageScore"]).reshape(-1, 1)
                gt_right = np.concatenate((right_landmarks, right_conf), axis=1)
            else:
                gt_right = np.zeros((21, 4))
                mpjpe_flag = False

        try:
            with open(pred_path) as pred_file:
                pred_json = json.load(pred_file)
        except FileNotFoundError:
            result_row["status"] = "missing_prediction"
            results.append(result_row)
            frame_num += capture_rate
            continue

        if pred_json.get("landmarks") is None:
            out_body = np.zeros((33, 3))
            mpjpe_flag = False
        else:
            # 保存されている33個のランドマークをインデックス順に並べる
            out_body = np.asarray([pred_json["landmarks"][str(idx)] for idx in range(33)], dtype=float)

        compe_gt, compe_pd = make_compe_landmarks(gt_body, out_body, body_map, confidence=confidence_threshold)
        left_compe_gt, left_compe_pd = make_compe_landmarks(gt_left, out_body, left_hand_map, confidence=confidence_threshold)
        right_compe_gt, right_compe_pd = make_compe_landmarks(gt_right, out_body, right_hand_map, confidence=confidence_threshold)

        if not mpjpe_flag:
            result_row["status"] = "missing_gt"
        else:
            if compe_gt.size > 0 and compe_pd.size > 0:
                result_row["body_mpjpe"] = float(compute_mpjpe(compe_pd, compe_gt[:, :3]))
            if left_compe_gt.size > 0 and left_compe_pd.size > 0:
                result_row["left_hand_mpjpe"] = float(compute_mpjpe(left_compe_pd, left_compe_gt[:, :3]))
            if right_compe_gt.size > 0 and right_compe_pd.size > 0:
                result_row["right_hand_mpjpe"] = float(compute_mpjpe(right_compe_pd, right_compe_gt[:, :3]))

            if (
                compe_gt.size > 0 and compe_pd.size > 0
                and left_compe_gt.size > 0 and left_compe_pd.size > 0
                and right_compe_gt.size > 0 and right_compe_pd.size > 0
            ):
                holistic_gt = np.concatenate((compe_gt, left_compe_gt, right_compe_gt), axis=0)
                holistic_pd = np.concatenate((compe_pd, left_compe_pd, right_compe_pd), axis=0)
                result_row["holistic_mpjpe"] = float(compute_mpjpe(holistic_pd, holistic_gt[:, :3]))
            else:
                result_row.setdefault("holistic_mpjpe", None)

        results.append(result_row)
        frame_num += capture_rate

    return results

if __name__ == "__main__": # テスト用
    results = run_mpjpe_evaluation(capture_rate=30)
    print("finished!!")
    for i in range(0, len(results)):
        if results[i]["status"] == 'ok':   
            print(f"{results[i]["right_hand_mpjpe"]:.2f}")
        else:
            print(results[i]['status'])
