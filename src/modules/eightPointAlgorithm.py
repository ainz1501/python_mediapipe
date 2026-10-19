import numpy as np


def eightPointAlgorithm(
    landmark1: np.ndarray,
    landmark2: np.ndarray,
    intrinsicParameter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return placeholder relative camera parameters."""
    rotationMatrix = np.eye(3, dtype=float)
    unitTranslation = np.array([1.0, 0.0, 0.0], dtype=float)
    return rotationMatrix, unitTranslation
