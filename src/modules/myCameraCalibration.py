import numpy as np

try:
    from . import depthFromDefocus as dfd
    from . import eightPointAlgorithm as epa
except ImportError:
    import depthFromDefocus as dfd
    import eightPointAlgorithm as epa


def recoverRealScaleTranslation(
    landmark1: np.ndarray,
    landmark2: np.ndarray,
    depthMap: np.ndarray,
    intrinsicParameter: np.ndarray,
    rotationMatrix: np.ndarray,
    unitTranslation: np.ndarray,
) -> np.ndarray:
    """Return a placeholder real-scale translation vector."""
    return np.asarray(unitTranslation, dtype=float).copy()


def myExtrinsicParameterCalibration(
    landmark1: np.ndarray,
    landmark2: np.ndarray,
    focalStack: dfd.FocalStack,
    intrinsicParameter: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate camera rotation and real-scale translation from two views."""
    if not isinstance(landmark1, np.ndarray):
        raise TypeError("landmark1 must be a numpy.ndarray")
    if not isinstance(landmark2, np.ndarray):
        raise TypeError("landmark2 must be a numpy.ndarray")
    if landmark1.ndim != 2 or landmark1.shape[1] != 2:
        raise ValueError("landmark1 must have shape (N, 2)")
    if landmark2.ndim != 2 or landmark2.shape[1] != 2:
        raise ValueError("landmark2 must have shape (N, 2)")
    if landmark1.shape[0] != landmark2.shape[0]:
        raise ValueError("landmark1 and landmark2 must contain the same number of points")
    if landmark1.shape[0] < 8:
        raise ValueError("at least 8 corresponding landmarks are required")
    if not isinstance(focalStack, dfd.FocalStack):
        raise TypeError("focalStack must be a depthFromDefocus.FocalStack instance")
    if not isinstance(intrinsicParameter, np.ndarray):
        raise TypeError("intrinsicParameter must be a numpy.ndarray")
    if intrinsicParameter.shape != (3, 3):
        raise ValueError("intrinsicParameter must have shape (3, 3)")

    rotationMatrix, unitTranslation = epa.eightPointAlgorithm(
        landmark1,
        landmark2,
        intrinsicParameter,
    )
    aifImage, depthMap = dfd.depthFromDefocus(focalStack)
    realScaleTranslation = recoverRealScaleTranslation(
        landmark1,
        landmark2,
        depthMap,
        intrinsicParameter,
        rotationMatrix,
        unitTranslation,
    )

    return rotationMatrix, realScaleTranslation

# テスト用
if __name__ == "__main__":
    landmark1 = np.array(
        [
            [120.0, 100.0],
            [220.0, 110.0],
            [320.0, 130.0],
            [420.0, 150.0],
            [140.0, 260.0],
            [240.0, 280.0],
            [340.0, 300.0],
            [440.0, 320.0],
        ],
        dtype=float,
    )
    # landmark1 = [[120.0, 100.0], [220.0, 110.0], [320.0, 130.0], [420.0, 150.0], [140.0, 260.0], [240.0, 280.0], [340.0, 300.0], [440.0, 320.0]]
    landmark2 = np.array(
        [
            [110.0, 100.0],
            [210.0, 110.0],
            [310.0, 130.0],
            [410.0, 150.0],
            [130.0, 260.0],
            [230.0, 280.0],
            [330.0, 300.0],
            [430.0, 320.0],
        ],
        dtype=float,
    )
    # landmark2 = [[110.0, 100.0], [210.0, 110.0], [310.0, 130.0], [410.0, 150.0], [130.0, 260.0], [230.0, 280.0], [330.0, 300.0], [430.0, 320.0]]
    focalStack = dfd.FocalStack()
    # focalStack = "test"
    intrinsicParameter = np.array(
        [
            [1000.0, 0.0, 320.0],
            [0.0, 1000.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    # intrinsicParameter = [[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]

    R, t = myExtrinsicParameterCalibration(
        landmark1,
        landmark2,
        focalStack,
        intrinsicParameter,
    )
    print(R,t)

""" テスト結果
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]] [1. 0. 0.]
"""
