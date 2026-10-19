import numpy as np


GOAL_MSE = 0.01
EPOCH = 40
NEAR_DEPTH = 0.1
FAR_DEPTH = 10.0
DEPTH_SAMPLING_NUM = 100
GOLDEN_SECTION_MAX_ITERATION = 100
GOLDEN_SECTION_TOLERANCE = 0.01
POWER_ITERATION_TIMES = 100
FISTA_INITIAL_ITERATION_TIMES = 200
FISTA_ITERATION_MAGNIFICATION = 1.05


class FocalStack:
    def __init__(
        self,
        img: np.ndarray,
        focalLength: float,
        focusDistance: np.ndarray,
        apertureDiameter: float,
        pixelSize: float,
    ) -> None:
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be np.ndarray")
        if img.ndim != 3:
            raise ValueError("img must have shape (K, HW, C)")
        if not isinstance(focalLength, float):
            raise TypeError("focalLength must be float")
        if not isinstance(focusDistance, np.ndarray):
            raise TypeError("focusDistance must be np.ndarray")
        if focusDistance.ndim != 1:
            raise ValueError("focusDistance must have shape (K,)")
        if not isinstance(apertureDiameter, float):
            raise TypeError("apertureDiameter must be float")
        if not isinstance(pixelSize, float):
            raise TypeError("pixelSize must be float")

        self.img = img.astype(float, copy=False)
        self.focalLength = focalLength
        self.focusDistance = focusDistance.astype(float, copy=False)
        self.apertureDiameter = apertureDiameter
        self.pixelSize = pixelSize
        self.allImg = self.makeAllImage()

    def makeAllImage(self) -> np.ndarray:
        allImage = np.empty((0, self.img.shape[2]), dtype=float)
        for i in range(self.img.shape[0]):
            allImage = np.vstack((allImage, self.img[i]))
        return allImage


def initAIFImage(focalStack: FocalStack) -> np.ndarray:
    """Return a placeholder initial all-in-focus image."""
    if not isinstance(focalStack, FocalStack):
        raise TypeError("focalStack must be FocalStack")
    return np.array(focalStack.img[0], dtype=float, copy=True)


def alternatingMinimization(
    initialAIFImage: np.ndarray,
    focalStack: FocalStack,
) -> tuple[np.ndarray, np.ndarray]:
    """Return placeholder optimized depth map and all-in-focus image."""
    if not isinstance(initialAIFImage, np.ndarray):
        raise TypeError("initialAIFImage must be np.ndarray")
    if not isinstance(focalStack, FocalStack):
        raise TypeError("focalStack must be FocalStack")

    depthMap = np.zeros(initialAIFImage.shape[0], dtype=float)
    AIFImage = np.array(initialAIFImage, dtype=float, copy=True)
    return depthMap, AIFImage


def depthFromDefocus(
    focalStack: FocalStack,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a depth map and all-in-focus image from a focal stack."""
    if not isinstance(focalStack, FocalStack):
        raise TypeError("focalStack must be FocalStack")

    initialAIFImage = initAIFImage(focalStack)
    depthMap, AIFImage = alternatingMinimization(initialAIFImage, focalStack)
    return depthMap, AIFImage


if __name__ == "__main__":
    validFocalStack = FocalStack(
        img=np.zeros((2, 4, 3), dtype=float),
        focalLength=0.05,
        focusDistance=np.array([1.0, 2.0], dtype=float),
        apertureDiameter=0.01,
        pixelSize=0.00001,
    )

    testDepthMap, testAIFImage = depthFromDefocus(validFocalStack)
    assert isinstance(testDepthMap, np.ndarray)
    assert isinstance(testAIFImage, np.ndarray)
    assert testDepthMap.shape == (4,)
    assert testAIFImage.shape == (4, 3)

    invalidFocalStacks = [
        None,
        np.zeros((2, 4, 3), dtype=float),
        "focalStack",
        1,
        1.0,
        [],
        {},
    ]
    for invalidFocalStack in invalidFocalStacks:
        try:
            depthFromDefocus(invalidFocalStack)
        except TypeError:
            pass
        else:
            raise AssertionError("depthFromDefocus must reject invalid focalStack type")
