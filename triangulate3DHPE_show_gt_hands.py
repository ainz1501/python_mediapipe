import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

"""GTと推定結果の手ランドマークを同一プロットへ可視化する補助スクリプト"""

# パラメータ設定
DATASET_NAME = "171204_pose3"
CAPTURE_RATE = 100
FIRST_FRAME_NUM = 137
END_FRAME_NUM = 9056
VIDEO1_NUM = 18
VIDEO2_NUM = 23

GT_HAND_DATA_FOLDER_PATH = Path("./panoptic-toolbox") / DATASET_NAME / "hdHand3d"
OUTPUT_LANDMARKS_PATH = Path("./outputs/output_landmarks") / f"{DATASET_NAME}_cam{VIDEO1_NUM:02d}{VIDEO2_NUM:02d}"

# CMU Panopticの手ランドマーク接続情報 (1始まり) を0始まりへ変換
GT_HAND_CONNECTIONS = np.array([
    [1, 2], [2, 3], [3, 4], [4, 5],
    [1, 6], [6, 7], [7, 8], [8, 9],
    [1, 10], [10, 11], [11, 12], [12, 13],
    [1, 14], [14, 15], [15, 16], [16, 17],
    [1, 18], [18, 19], [19, 20], [20, 21],
], dtype=int) - 1

# triangulate3DHPE_compute_mpjpe copy.py から引用した対応表
RIGHT_HAND_MAP = np.array([
    [0, 16],  # Wrist
    [17, 18],  # Pinky fingertip
    [5, 20],   # Index fingertip
    [2, 22],   # Thumb tip
], dtype=int)

LEFT_HAND_MAP = np.array([
    [0, 15],  # Wrist
    [17, 17],  # Pinky fingertip
    [5, 19],   # Index fingertip
    [2, 21],   # Thumb tip
], dtype=int)


def load_gt_hand_landmarks(frame_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """パンオプティックデータセットから指定フレームの左右手ランドマークを取得する"""
    json_path = GT_HAND_DATA_FOLDER_PATH / f"handRecon3D_hd{frame_num:08d}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Hand GT file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        frame_data = json.load(f)

    if not frame_data.get("people"):
        return np.zeros((21, 3)), np.zeros((21, 3))

    person = frame_data["people"][0]

    left_hand = person.get("left_hand", {})
    right_hand = person.get("right_hand", {})

    left = np.array(left_hand.get("landmarks", []), dtype=float).reshape(-1, 3) if left_hand else np.zeros((21, 3))
    right = np.array(right_hand.get("landmarks", []), dtype=float).reshape(-1, 3) if right_hand else np.zeros((21, 3))

    if left.size == 0:
        left = np.zeros((21, 3))
    if right.size == 0:
        right = np.zeros((21, 3))

    return left, right


def load_predicted_landmarks(frame_num: int) -> np.ndarray:
    """triangulate3DHPEで保存した33点の3Dランドマークを読み込む"""
    json_path = OUTPUT_LANDMARKS_PATH / f"frame_{frame_num:08d}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Predicted landmark file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    landmarks = data.get("landmarks")
    if not landmarks:
        return np.zeros((33, 3))

    coords = np.zeros((33, 3), dtype=float)
    for idx in range(33):
        coords[idx] = landmarks.get(str(idx), [0.0, 0.0, 0.0])

    return coords


def extract_hand_from_predictions(predicted: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """対応表を用いて推定結果から手領域のみを抽出 (欠損は0埋め)"""
    hand = np.zeros((21, 3))
    if predicted.size == 0:
        return hand

    max_pd_index = predicted.shape[0] - 1
    for gt_idx, pd_idx in mapping:
        if 0 <= gt_idx < hand.shape[0] and 0 <= pd_idx <= max_pd_index:
            hand[gt_idx] = predicted[pd_idx]

    return hand


def _draw_hand(
    ax: plt.Axes,
    landmarks: np.ndarray,
    connections: Optional[np.ndarray],
    color: str,
    label: str,
    marker: str = "o",
    highlight_indices: Optional[np.ndarray] = None,
    highlight_color: str = "red",
    highlight_label: Optional[str] = None,
) -> None:
    """Landmarksと骨格線を描画し、指定されたランドマークを強調表示"""
    valid_mask = ~np.all(landmarks == 0, axis=1)
    valid_points = landmarks[valid_mask]

    if valid_points.size == 0:
        ax.text(0.0, 0.0, 0.0, f"{label}: no data", color=color)
        return

    ax.scatter(valid_points[:, 0], valid_points[:, 1], valid_points[:, 2], s=30, c=color, marker=marker, label=label)

    if connections is not None:
        for start_idx, end_idx in connections:
            if valid_mask[start_idx] and valid_mask[end_idx]:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                ax.plot(
                    [start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    c=color,
                    linewidth=2,
                )

    if highlight_indices is not None:
        indices = np.unique(np.asarray(highlight_indices, dtype=int))
        valid_indices = [idx for idx in indices if 0 <= idx < landmarks.shape[0] and valid_mask[idx]]
        if valid_indices:
            highlight_points = landmarks[valid_indices]
            ax.scatter(
                highlight_points[:, 0],
                highlight_points[:, 1],
                highlight_points[:, 2],
                s=60,
                c=highlight_color,
                marker="o",
                label=highlight_label,
            )


def _set_equal_aspect(ax: plt.Axes, points: np.ndarray) -> None:
    """3D座標を等方スケールで表示"""
    if points.size == 0:
        return

    max_range = (points.max(axis=0) - points.min(axis=0)).max()
    mid = points.mean(axis=0)
    half = max_range / 2
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.set_box_aspect([1, 1, 1])


def visualize_gt_hands(frame_num: int) -> None:
    """単一フレームの左右手GTと推定結果を表示"""
    left_gt, right_gt = load_gt_hand_landmarks(frame_num)
    predicted = load_predicted_landmarks(frame_num)
    left_pred = extract_hand_from_predictions(predicted, LEFT_HAND_MAP)
    right_pred = extract_hand_from_predictions(predicted, RIGHT_HAND_MAP)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    _draw_hand(
        ax,
        left_gt,
        GT_HAND_CONNECTIONS,
        color="tab:blue",
        label="GT left",
        highlight_indices=LEFT_HAND_MAP[:, 0],
        highlight_label="GT left (matched)",
    )
    _draw_hand(
        ax,
        right_gt,
        GT_HAND_CONNECTIONS,
        color="tab:orange",
        label="GT right",
        highlight_indices=RIGHT_HAND_MAP[:, 0],
        highlight_label="GT right (matched)",
    )
    _draw_hand(
        ax,
        left_pred,
        GT_HAND_CONNECTIONS,
        color="tab:purple",
        label="Pred left",
        marker="^",
    )
    _draw_hand(
        ax,
        right_pred,
        GT_HAND_CONNECTIONS,
        color="tab:green",
        label="Pred right",
        marker="s",
    )

    stacked = np.vstack([left_gt, right_gt, left_pred, right_pred])
    stacked = stacked[~np.all(stacked == 0, axis=1)]
    if stacked.size:
        _set_equal_aspect(ax, stacked)

    ax.set_title(f"Hands (GT vs Pred) frame {frame_num:08d}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def main() -> None:
    for frame_num in range(FIRST_FRAME_NUM, END_FRAME_NUM, CAPTURE_RATE):
        try:
            visualize_gt_hands(frame_num)
        except FileNotFoundError as exc:
            print(exc)
            break

        user_input = input("Press Enter for next frame or 'q' to quit: ")
        if user_input.strip().lower() == "q":
            break


if __name__ == '__main__':
    main()
