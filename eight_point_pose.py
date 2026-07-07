"""
Estimate relative camera pose from 2D correspondences using eight-point
algorithm variants.

This module intentionally depends only on NumPy. It is meant to be easy to
import from experiments and from triangulate3DHPE.py without triggering image
loading, MediaPipe initialization, or dataset access.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

PointArray = np.ndarray
Matrix3x3 = np.ndarray
PoseCandidate = tuple[np.ndarray, np.ndarray]


def _Validate_points(points_left: PointArray, points_right: PointArray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert corresponding image points."""
    left = np.asarray(points_left, dtype=float)
    right = np.asarray(points_right, dtype=float)

    if left.ndim != 2 or left.shape[1] != 2:
        raise ValueError(f"points_left must have shape (N, 2): {left.shape}")
    if right.ndim != 2 or right.shape[1] != 2:
        raise ValueError(f"points_right must have shape (N, 2): {right.shape}")
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"point counts must match: {left.shape[0]} != {right.shape[0]}")
    if left.shape[0] < 8:
        raise ValueError(f"at least 8 point correspondences are required: {left.shape[0]}")

    return left, right


def _Validate_intrinsic_matrix(K: Matrix3x3, name: str) -> np.ndarray:
    """Validate and convert a camera intrinsic matrix."""
    matrix = np.asarray(K, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3): {matrix.shape}")
    return matrix


def _As_homogeneous(points: PointArray) -> np.ndarray:
    """Convert shape (N, 2) points to shape (N, 3) homogeneous points."""
    return np.column_stack((points, np.ones(points.shape[0])))


def Normalize_points(points: PointArray) -> tuple[PointArray, Matrix3x3]:
    """
    Normalize 2D points for the eight-point algorithm.

    Args:
        points: Image points with shape `(N, 2)`.

    Returns:
        `(normalized_points, T)`.
        `normalized_points` has shape `(N, 2)`.
        `T` is a shape `(3, 3)` transform matrix such that
        `p_normalized = T @ [x, y, 1]`.

    Notes:
        The transform moves the centroid to `(0, 0)` and scales points so that
        the root mean square distance from the origin is `sqrt(2)`. This is the
        standard normalization used by the normalized eight-point algorithm and
        makes the linear SVD solve numerically better conditioned.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must have shape (N, 2): {points.shape}")

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    rms_distance = np.sqrt(np.mean(np.sum(centered * centered, axis=1)))
    if rms_distance <= np.finfo(float).eps:
        raise ValueError("points are degenerate: all points are too close to the centroid")

    scale = np.sqrt(2.0) / rms_distance
    T = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    normalized_h = (T @ _As_homogeneous(points).T).T
    return normalized_h[:, :2], T


def _Enforce_rank2(matrix: Matrix3x3) -> Matrix3x3:
    """Project a 3x3 matrix onto the closest rank-2 matrix by SVD."""
    U, singular_values, Vt = np.linalg.svd(matrix)
    singular_values[-1] = 0.0
    return U @ np.diag(singular_values) @ Vt


def _Enforce_essential_constraints(E: Matrix3x3) -> Matrix3x3:
    """Project a 3x3 matrix onto the essential-matrix constraint space."""
    U, singular_values, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0

    average = 0.5 * (singular_values[0] + singular_values[1])
    return U @ np.diag([average, average, 0.0]) @ Vt


def Estimate_fundamental_matrix(points_left: PointArray, points_right: PointArray) -> Matrix3x3:
    """
    Estimate the fundamental matrix by the normalized eight-point algorithm.

    Args:
        points_left: Left image points with shape `(N, 2)`.
        points_right: Right image points with shape `(N, 2)`.

    Returns:
        Fundamental matrix `F` with shape `(3, 3)`.
        It satisfies `p_right.T @ F @ p_left = 0`.

    Raises:
        ValueError: If there are fewer than 8 correspondences, point counts do
            not match, input shapes are invalid, or the linear solve is
            degenerate.
    """
    left, right = _Validate_points(points_left, points_right)
    left_norm, T_left = Normalize_points(left)
    right_norm, T_right = Normalize_points(right)

    xl = left_norm[:, 0]
    yl = left_norm[:, 1]
    xr = right_norm[:, 0]
    yr = right_norm[:, 1]
    A = np.column_stack(
        (
            xr * xl,
            xr * yl,
            xr,
            yr * xl,
            yr * yl,
            yr,
            xl,
            yl,
            np.ones(left_norm.shape[0]),
        )
    )

    _, singular_values, Vt = np.linalg.svd(A)
    if singular_values[-1] <= np.finfo(float).eps:
        # Exact zero can happen for ideal data, so this is not necessarily bad.
        pass
    if Vt.shape[0] < 9:
        raise ValueError("failed to solve the eight-point linear system")

    F_norm = Vt[-1].reshape(3, 3)
    F_norm = _Enforce_rank2(F_norm)
    F = T_right.T @ F_norm @ T_left

    norm = np.linalg.norm(F)
    if norm <= np.finfo(float).eps:
        raise ValueError("estimated fundamental matrix is degenerate")
    return F / norm


def Estimate_fundamental_matrix_raw(points_left: PointArray, points_right: PointArray) -> Matrix3x3:
    """
    Estimate the fundamental matrix without point normalization or rank correction.

    Args:
        points_left: Left image points with shape `(N, 2)`.
        points_right: Right image points with shape `(N, 2)`.

    Returns:
        Raw fundamental matrix `F` with shape `(3, 3)`.
        It is the direct reshape of `Vt[-1]` from solving `A f = 0`.

    Notes:
        This is intentionally the minimal eight-point algorithm for comparison
        and learning. It does not normalize 2D points and does not enforce
        `rank(F) = 2`.
    """
    left, right = _Validate_points(points_left, points_right)

    xl = left[:, 0]
    yl = left[:, 1]
    xr = right[:, 0]
    yr = right[:, 1]
    A = np.column_stack(
        (
            xr * xl,
            xr * yl,
            xr,
            yr * xl,
            yr * yl,
            yr,
            xl,
            yl,
            np.ones(left.shape[0]),
        )
    )

    _, _, Vt = np.linalg.svd(A)
    if Vt.shape[0] < 9:
        raise ValueError("failed to solve the raw eight-point linear system")

    F = Vt[-1].reshape(3, 3)
    if np.linalg.norm(F) <= np.finfo(float).eps:
        raise ValueError("estimated raw fundamental matrix is degenerate")
    return F


def Estimate_essential_matrix(
    points_left: PointArray,
    points_right: PointArray,
    K_left: Matrix3x3,
    K_right: Matrix3x3,
) -> tuple[Matrix3x3, Matrix3x3]:
    """
    Estimate the essential matrix from image correspondences and intrinsics.

    Args:
        points_left: Left image points with shape `(N, 2)`.
        points_right: Right image points with shape `(N, 2)`.
        K_left: Left camera intrinsic matrix with shape `(3, 3)`.
        K_right: Right camera intrinsic matrix with shape `(3, 3)`.

    Returns:
        `(E, F)`.
        `E` is the corrected essential matrix with singular values `(s, s, 0)`.
        `F` is the corrected rank-2 fundamental matrix.
    """
    K_left = _Validate_intrinsic_matrix(K_left, "K_left")
    K_right = _Validate_intrinsic_matrix(K_right, "K_right")

    F = Estimate_fundamental_matrix(points_left, points_right)
    E = K_right.T @ F @ K_left
    E = _Enforce_essential_constraints(E)
    return E, F


def Estimate_essential_matrix_raw(
    points_left: PointArray,
    points_right: PointArray,
    K_left: Matrix3x3,
    K_right: Matrix3x3,
) -> tuple[Matrix3x3, Matrix3x3]:
    """
    Estimate the essential matrix without normalization or singular-value correction.

    Args:
        points_left: Left image points with shape `(N, 2)`.
        points_right: Right image points with shape `(N, 2)`.
        K_left: Left camera intrinsic matrix with shape `(3, 3)`.
        K_right: Right camera intrinsic matrix with shape `(3, 3)`.

    Returns:
        `(E, F)`.
        `F` is estimated by `Estimate_fundamental_matrix_raw`.
        `E` is computed as `K_right.T @ F @ K_left` without enforcing
        singular values `(s, s, 0)`.
    """
    K_left = _Validate_intrinsic_matrix(K_left, "K_left")
    K_right = _Validate_intrinsic_matrix(K_right, "K_right")

    F = Estimate_fundamental_matrix_raw(points_left, points_right)
    E = K_right.T @ F @ K_left
    return E, F


def Decompose_essential_matrix(E: Matrix3x3) -> list[PoseCandidate]:
    """
    Decompose an essential matrix into four possible relative poses.

    Args:
        E: Essential matrix with shape `(3, 3)`.

    Returns:
        A list of four `(R, t)` candidates.
        `R` has shape `(3, 3)`.
        `t` has shape `(3,)` and represents translation direction only.
    """
    E = np.asarray(E, dtype=float)
    if E.shape != (3, 3):
        raise ValueError(f"E must have shape (3, 3): {E.shape}")

    U, _, Vt = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1.0
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1.0

    W = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    if np.linalg.det(R1) < 0:
        R1 *= -1.0
    if np.linalg.det(R2) < 0:
        R2 *= -1.0

    t = U[:, 2]
    t_norm = np.linalg.norm(t)
    if t_norm <= np.finfo(float).eps:
        raise ValueError("essential matrix decomposition produced a degenerate translation")
    t = t / t_norm

    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]


def Decompose_essential_matrix_raw(E: Matrix3x3) -> list[PoseCandidate]:
    """
    Decompose an uncorrected essential matrix into four relative-pose candidates.

    Args:
        E: Raw essential matrix with shape `(3, 3)`.

    Returns:
        A list of four `(R, t)` candidates.

    Notes:
        This function does not modify the singular values of `E`. It only uses
        SVD to extract pose candidates and fixes determinant signs so returned
        rotations are proper rotation matrices when possible.
    """
    return Decompose_essential_matrix(E)


def _Triangulate_point(P_left: np.ndarray, P_right: np.ndarray, point_left: np.ndarray, point_right: np.ndarray) -> np.ndarray:
    """Triangulate one 3D point from two image observations."""
    xl, yl = point_left
    xr, yr = point_right
    A = np.array(
        [
            xl * P_left[2] - P_left[0],
            yl * P_left[2] - P_left[1],
            xr * P_right[2] - P_right[0],
            yr * P_right[2] - P_right[1],
        ],
        dtype=float,
    )
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    if abs(X_h[3]) <= np.finfo(float).eps:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return X_h[:3] / X_h[3]


def _Triangulate_points_for_pose(
    points_left: PointArray,
    points_right: PointArray,
    K_left: Matrix3x3,
    K_right: Matrix3x3,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """Triangulate all correspondences for one relative pose."""
    P_left = K_left @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P_right = K_right @ np.hstack((R, t.reshape(3, 1)))
    points3d = [
        _Triangulate_point(P_left, P_right, point_left, point_right)
        for point_left, point_right in zip(points_left, points_right)
    ]
    return np.asarray(points3d, dtype=float)


def Choose_correct_pose(
    pose_candidates: Sequence[PoseCandidate],
    points_left: PointArray,
    points_right: PointArray,
    K_left: Matrix3x3,
    K_right: Matrix3x3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Choose the pose candidate with the most points in front of both cameras.

    Args:
        pose_candidates: Candidate `(R, t)` pairs from `Decompose_essential_matrix`.
        points_left: Left image points with shape `(N, 2)`.
        points_right: Right image points with shape `(N, 2)`.
        K_left: Left camera intrinsic matrix with shape `(3, 3)`.
        K_right: Right camera intrinsic matrix with shape `(3, 3)`.

    Returns:
        `(R, t, points3d, positive_depth_count)` for the best candidate.

    Notes:
        `t` is still scale-free. Cheirality only selects its sign and the
        correct rotation candidate.
    """
    left, right = _Validate_points(points_left, points_right)
    K_left = _Validate_intrinsic_matrix(K_left, "K_left")
    K_right = _Validate_intrinsic_matrix(K_right, "K_right")

    if len(pose_candidates) == 0:
        raise ValueError("pose_candidates must not be empty")

    best_result: tuple[np.ndarray, np.ndarray, np.ndarray, int] | None = None
    best_count = -1

    for R, t in pose_candidates:
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3)
        if R.shape != (3, 3):
            raise ValueError(f"candidate R must have shape (3, 3): {R.shape}")

        points3d = _Triangulate_points_for_pose(left, right, K_left, K_right, R, t)
        finite_mask = np.isfinite(points3d).all(axis=1)
        depth_left = points3d[:, 2]
        points_right_camera = (R @ points3d.T + t.reshape(3, 1)).T
        depth_right = points_right_camera[:, 2]
        positive_mask = finite_mask & (depth_left > 0.0) & (depth_right > 0.0)
        count = int(np.count_nonzero(positive_mask))

        if count > best_count:
            best_count = count
            best_result = (R, t, points3d, count)

    if best_result is None:
        raise ValueError("failed to choose a valid pose candidate")
    return best_result


def Estimate_relative_pose_eight_point(
    points_left: PointArray,
    points_right: PointArray,
    K_left: Matrix3x3,
    K_right: Matrix3x3,
) -> dict[str, Any]:
    """
    Estimate relative camera pose by the normalized eight-point algorithm.

    Args:
        points_left: Left image points with shape `(N, 2)`.
        points_right: Right image points with shape `(N, 2)`.
        K_left: Left camera intrinsic matrix with shape `(3, 3)`.
        K_right: Right camera intrinsic matrix with shape `(3, 3)`.

    Returns:
        A dictionary containing:
        `R`: Selected relative rotation from left camera to right camera.
        `t`: Selected relative translation direction from left to right.
        `E`: Corrected essential matrix.
        `F`: Corrected fundamental matrix.
        `points3d`: Triangulated points for the selected pose.
        `positive_depth_count`: Number of points in front of both cameras.
        `pose_candidates`: The four decomposed pose candidates.

    Raises:
        ValueError: If inputs are invalid or the estimation degenerates.
    """
    left, right = _Validate_points(points_left, points_right)
    K_left = _Validate_intrinsic_matrix(K_left, "K_left")
    K_right = _Validate_intrinsic_matrix(K_right, "K_right")

    E, F = Estimate_essential_matrix(left, right, K_left, K_right)
    pose_candidates = Decompose_essential_matrix(E)
    R, t, points3d, positive_depth_count = Choose_correct_pose(
        pose_candidates,
        left,
        right,
        K_left,
        K_right,
    )

    return {
        "R": R,
        "t": t,
        "E": E,
        "F": F,
        "points3d": points3d,
        "positive_depth_count": positive_depth_count,
        "pose_candidates": pose_candidates,
    }


def Estimate_relative_pose_eight_point_raw(
    points_left: PointArray,
    points_right: PointArray,
    K_left: Matrix3x3,
    K_right: Matrix3x3,
) -> dict[str, Any]:
    """
    Estimate relative pose with the raw eight-point algorithm.

    Args:
        points_left: Left image points with shape `(N, 2)`.
        points_right: Right image points with shape `(N, 2)`.
        K_left: Left camera intrinsic matrix with shape `(3, 3)`.
        K_right: Right camera intrinsic matrix with shape `(3, 3)`.

    Returns:
        A dictionary containing:
        `R`: Selected relative rotation from left camera to right camera.
        `t`: Selected relative translation direction from left to right.
        `E`: Raw essential matrix, without singular-value correction.
        `F`: Raw fundamental matrix, without normalization or rank correction.
        `points3d`: Triangulated points for the selected pose.
        `positive_depth_count`: Number of points in front of both cameras.
        `pose_candidates`: The four decomposed pose candidates.
        `F_rank`: Numeric rank of raw `F`.
        `E_singular_values`: Singular values of raw `E`.
        `det_R`: Determinant of the selected rotation matrix.
        `rotation_orthogonality_error`: Frobenius norm of `R.T @ R - I`.
    """
    left, right = _Validate_points(points_left, points_right)
    K_left = _Validate_intrinsic_matrix(K_left, "K_left")
    K_right = _Validate_intrinsic_matrix(K_right, "K_right")

    E, F = Estimate_essential_matrix_raw(left, right, K_left, K_right)
    pose_candidates = Decompose_essential_matrix_raw(E)
    R, t, points3d, positive_depth_count = Choose_correct_pose(
        pose_candidates,
        left,
        right,
        K_left,
        K_right,
    )
    E_singular_values = np.linalg.svd(E, compute_uv=False)

    return {
        "R": R,
        "t": t,
        "E": E,
        "F": F,
        "points3d": points3d,
        "positive_depth_count": positive_depth_count,
        "pose_candidates": pose_candidates,
        "F_rank": int(np.linalg.matrix_rank(F)),
        "E_singular_values": E_singular_values,
        "det_R": float(np.linalg.det(R)),
        "rotation_orthogonality_error": float(np.linalg.norm(R.T @ R - np.eye(3))),
    }
