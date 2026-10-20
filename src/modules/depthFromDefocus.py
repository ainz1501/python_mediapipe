from pprint import pformat

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


def _validateFocalStackValues(
    img: np.ndarray,
    focalLength: float,
    focusDistance: np.ndarray,
    apertureDiameter: float,
    pixelSize: float,
) -> None:
    if np.any(img < 0.0) or np.any(img > 255.0):
        raise ValueError("img values must be between 0 and 255")
    if focalLength <= 0.0:
        raise ValueError("focalLength must be greater than 0")
    if np.any(focusDistance <= 0.0):
        raise ValueError("focusDistance values must be greater than 0")
    if apertureDiameter <= 0.0:
        raise ValueError("apertureDiameter must be greater than 0")
    if pixelSize <= 0.0:
        raise ValueError("pixelSize must be greater than 0")
    if img.shape[0] != focusDistance.shape[0]:
        raise ValueError("img count must match focusDistance count")


class FocalStack:
    def __init__(
        self,
        img: np.ndarray,
        focalLength: float,
        focusDistance: np.ndarray,
        apertureDiameter: float,
        pixelSize: float,
    ):
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be np.ndarray")
        if not isinstance(focalLength, float):
            raise TypeError("focalLength must be float")
        if not isinstance(focusDistance, np.ndarray):
            raise TypeError("focusDistance must be np.ndarray")
        if not isinstance(apertureDiameter, float):
            raise TypeError("apertureDiameter must be float")
        if not isinstance(pixelSize, float):
            raise TypeError("pixelSize must be float")
        _validateFocalStackValues(
            img,
            focalLength,
            focusDistance,
            apertureDiameter,
            pixelSize,
        )

        self.img = img
        self.focalLength = focalLength
        self.focusDistance = focusDistance
        self.apertureDiameter = apertureDiameter
        self.pixelSize = pixelSize
        self.allImg = self.makeAllImage()

    def makeAllImage(self) -> np.ndarray:
        return np.concatenate(self.img, axis=0)


def initAIFImage(focalStack: FocalStack) -> np.ndarray:
    """Return a minimal placeholder initial AIF image."""
    return np.empty((0, 0), dtype=float)


def alternatingMinimization(
    initialAIFImage: np.ndarray,
    focalStack: FocalStack,
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimal placeholder depth and AIF arrays."""
    depthMap = np.empty((0, 0), dtype=float)
    AIFImage = np.empty((0, 0), dtype=float)
    return depthMap, AIFImage


def depthFromDefocus(
    focalStack: FocalStack,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a depth map and all-in-focus image from a focal stack."""
    if not isinstance(focalStack, FocalStack):
        raise TypeError("focalStack must be FocalStack")
    _validateFocalStackValues(
        focalStack.img,
        focalStack.focalLength,
        focalStack.focusDistance,
        focalStack.apertureDiameter,
        focalStack.pixelSize,
    )

    initialAIFImage = initAIFImage(focalStack)
    depthMap, AIFImage = alternatingMinimization(initialAIFImage, focalStack)
    return depthMap, AIFImage


if __name__ == "__main__":
    # 条件が偽の場合にテストを不合格として停止する共通確認関数。
    def require(condition, message):
        if not condition:
            raise AssertionError(message)

    # 各テストの説明と結果を同じ形式で表示し、不合格時は例外を再送出して後続テストを停止する。
    def runTestCase(
        category,
        number,
        targetFunction,
        name,
        overview,
        testInput,
        executionCode,
        goal,
        expectedOutput,
        testFunction,
    ):
        print(f"\n[{category} {number}] {name}")
        print(f"対象関数 : {targetFunction}")
        print(f"概要     : {overview}")
        print(f"入力     : {testInput}")
        print(f"実行コード: {executionCode}")
        print(f"期待条件 : {goal}")
        print(f"期待出力 : {pformat(expectedOutput, sort_dicts=False)}")
        try:
            actualOutput = testFunction()
        except Exception as error:
            print(f"実際の出力: {type(error).__name__}: {error}")
            print("結果     : 不合格")
            raise
        print(f"実際の出力: {pformat(actualOutput, sort_dicts=False)}")
        if actualOutput != expectedOutput:
            print("出力比較 : 不一致")
            print("結果     : 不合格")
            raise AssertionError("実際の出力が期待出力と一致しません")
        print("出力比較 : 一致")
        print("結果     : 合格")

    # 正常系テストで全関数へ渡す、数値条件も満たしたFocalStack。
    validFocalStack = FocalStack(
        img=np.array(
            [
                [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                [[70.0, 80.0, 90.0], [100.0, 110.0, 120.0]],
            ]
        ),
        focalLength=0.05,
        focusDistance=np.array([1.0, 2.0]),
        apertureDiameter=0.01,
        pixelSize=0.000005,
    )
    testCounts = {
        "個別テスト": 0,
        "統合テスト": 0,
        "異常系テスト": 0,
    }

    # 個別テスト1: initAIFImageの出力契約
    def testInitAIFImage():
        # 仮置き関数を実行し、仕様どおり画像を表す2次元配列が返ることを確認する。
        output = initAIFImage(validFocalStack)
        require(isinstance(output, np.ndarray), "出力はnp.ndarrayである必要があります")
        require(output.ndim == 2, "出力は2次元である必要があります")
        # ターミナル表示用に、実際の型とshapeを返す。
        return {"型": type(output).__name__, "shape": output.shape}

    testCounts["個別テスト"] += 1
    runTestCase(
        "個別テスト",
        testCounts["個別テスト"],
        "initAIFImage",
        "initAIFImageの出力契約",
        "仮置き関数が返す値の型と次元を確認します。",
        "FocalStackのインスタンス",
        "output = initAIFImage(validFocalStack)",
        "2次元のnp.ndarrayを返す",
        {"型": "ndarray", "shape": (0, 0)},
        testInitAIFImage,
    )

    # 個別テスト2: alternatingMinimizationの出力契約
    def testAlternatingMinimization():
        # 初期AIF画像として、仕様上必要な最小の2次元配列を用意する。
        initialAIFImage = np.empty((0, 0), dtype=float)
        # 仮置き関数を実行し、深度マップとAIF画像の2つの戻り値を取得する。
        depthMap, AIFImage = alternatingMinimization(
            initialAIFImage,
            validFocalStack,
        )
        # 深度マップは各画素位置を保持する2次元配列であることを確認する。
        require(isinstance(depthMap, np.ndarray), "depthMapはnp.ndarrayである必要があります")
        require(depthMap.ndim == 2, "depthMapは2次元である必要があります")
        # AIF画像も画像データを保持する2次元配列であることを確認する。
        require(isinstance(AIFImage, np.ndarray), "AIFImageはnp.ndarrayである必要があります")
        require(AIFImage.ndim == 2, "AIFImageは2次元である必要があります")
        return {
            "depthMap": {"型": type(depthMap).__name__, "shape": depthMap.shape},
            "AIFImage": {"型": type(AIFImage).__name__, "shape": AIFImage.shape},
        }

    testCounts["個別テスト"] += 1
    runTestCase(
        "個別テスト",
        testCounts["個別テスト"],
        "alternatingMinimization",
        "alternatingMinimizationの出力契約",
        "仮置き関数の戻り値数、型、次元を確認します。",
        "initialAIFImageのshape=(0, 0)、FocalStackのインスタンス",
        "depthMap, AIFImage = alternatingMinimization(initialAIFImage, validFocalStack)",
        "2次元のnp.ndarrayを2つ返す",
        {
            "depthMap": {"型": "ndarray", "shape": (0, 0)},
            "AIFImage": {"型": "ndarray", "shape": (0, 0)},
        },
        testAlternatingMinimization,
    )

    # 統合テスト1: depthFromDefocusの実関数による入出力
    def testIntegratedFlow():
        # 実際の関数へ既知のFocalStackを入力し、戻り値を直接取得する。
        actualDepthMap, actualAIFImage = depthFromDefocus(validFocalStack)
        # 現在の仮置き実装が返す空の2次元配列と一致することを確認する。
        expectedDepthMap = np.empty((0, 0), dtype=float)
        expectedAIFImage = np.empty((0, 0), dtype=float)
        np.testing.assert_array_equal(actualDepthMap, expectedDepthMap)
        np.testing.assert_array_equal(actualAIFImage, expectedAIFImage)
        # 実際の戻り値が深度と画像値の不変条件を満たすことを確認する。
        require(np.all(actualDepthMap >= 0.0), "depthMapに負の値が含まれています")
        require(np.all(actualAIFImage >= 0.0), "AIFImageに負の値が含まれています")
        require(np.all(actualAIFImage <= 255.0), "AIFImageに255を超える値が含まれています")
        return {
            "depthMapの型": type(actualDepthMap).__name__,
            "depthMapのshape": actualDepthMap.shape,
            "depthMap": actualDepthMap.tolist(),
            "AIFImageの型": type(actualAIFImage).__name__,
            "AIFImageのshape": actualAIFImage.shape,
            "AIFImage": actualAIFImage.tolist(),
        }

    testCounts["統合テスト"] += 1
    runTestCase(
        "統合テスト",
        testCounts["統合テスト"],
        "depthFromDefocus",
        "depthFromDefocusの実関数による入出力",
        "実際の関数へ既知入力を渡し、戻り値と不変条件を確認します。",
        "数値条件を満たすFocalStackのインスタンス",
        "actualDepthMap, actualAIFImage = depthFromDefocus(validFocalStack)",
        "空の2次元配列を2つ返し、深度>=0、0<=画像値<=255",
        {
            "depthMapの型": "ndarray",
            "depthMapのshape": (0, 0),
            "depthMap": [],
            "AIFImageの型": "ndarray",
            "AIFImageのshape": (0, 0),
            "AIFImage": [],
        },
        testIntegratedFlow,
    )

    # 異常系テスト1〜2: nullと不適切な型に対する型エラー
    invalidFocalStacks = [
        ("null（None）入力", None, "None"),  # 異常系テスト1: null
        ("不適切なstr型入力", "focalStack", "'focalStack'"),  # 異常系テスト2: 不適切な型
    ]

    for inputName, invalidFocalStack, inputCode in invalidFocalStacks:
        # 上記リストの1要素を、独立した1テストケースとして実行する。
        def testInvalidType(value=invalidFocalStack):
            try:
                # 不正な型を入力し、対象関数が処理を続けずTypeErrorを送出することを期待する。
                depthFromDefocus(value)
            except TypeError as error:
                # 期待したTypeErrorを、実際の出力としてターミナルへ表示する。
                return f"TypeError: {error}"
            # 例外が発生しなかった場合は、型検証が機能していないため不合格とする。
            raise AssertionError("TypeErrorが発生しませんでした")

        testCounts["異常系テスト"] += 1
        runTestCase(
            "異常系テスト",
            testCounts["異常系テスト"],
            "depthFromDefocus",
            f"depthFromDefocusが{inputName}を拒否すること",
            "不正な入力型に対する処理を確認します。",
            repr(invalidFocalStack),
            f"depthFromDefocus({inputCode})",
            "TypeErrorが発生する",
            "TypeError: focalStack must be FocalStack",
            testInvalidType,
        )

    # 異常系テスト3〜9: depthFromDefocusの数値条件に対する値エラー
    invalidNumericConditions = [
        (
            "画素値が0未満",
            "img",
            np.array([[[-1.0, 20.0, 30.0]], [[40.0, 50.0, 60.0]]]),
            "img values must be between 0 and 255",
        ),
        (
            "画素値が255超過",
            "img",
            np.array([[[256.0, 20.0, 30.0]], [[40.0, 50.0, 60.0]]]),
            "img values must be between 0 and 255",
        ),
        (
            "焦点距離が0以下",
            "focalLength",
            0.0,
            "focalLength must be greater than 0",
        ),
        (
            "合焦距離に0以下を含む",
            "focusDistance",
            np.array([1.0, 0.0]),
            "focusDistance values must be greater than 0",
        ),
        (
            "レンズ開口径が0以下",
            "apertureDiameter",
            0.0,
            "apertureDiameter must be greater than 0",
        ),
        (
            "ピクセルサイズが0以下",
            "pixelSize",
            0.0,
            "pixelSize must be greater than 0",
        ),
        (
            "画像枚数と合焦距離数が不一致",
            "focusDistance",
            np.array([1.0]),
            "img count must match focusDistance count",
        ),
    ]

    for conditionName, argumentName, invalidValue, expectedMessage in invalidNumericConditions:
        # 正常な引数のうち1項目だけを異常値へ置き換え、原因を限定する。
        def testInvalidNumericValue(
            name=argumentName,
            value=invalidValue,
            message=expectedMessage,
        ):
            # 各テストで独立した正常入力を作り、対象のプロパティだけを異常値へ変更する。
            invalidStack = FocalStack(
                img=validFocalStack.img.copy(),
                focalLength=validFocalStack.focalLength,
                focusDistance=validFocalStack.focusDistance.copy(),
                apertureDiameter=validFocalStack.apertureDiameter,
                pixelSize=validFocalStack.pixelSize,
            )
            setattr(invalidStack, name, value)
            try:
                # 数値条件に違反した入力を実関数へ渡し、ValueErrorが発生することを確認する。
                depthFromDefocus(invalidStack)
            except ValueError as error:
                require(str(error) == message, "ValueErrorのメッセージが期待値と異なります")
                return f"ValueError: {error}"
            raise AssertionError("ValueErrorが発生しませんでした")

        testCounts["異常系テスト"] += 1
        runTestCase(
            "異常系テスト",
            testCounts["異常系テスト"],
            "depthFromDefocus",
            f"depthFromDefocusが{conditionName}を拒否すること",
            "光学値または画像値の数値条件に対する処理を確認します。",
            f"{argumentName}={pformat(invalidValue)}",
            "depthFromDefocus(invalidStack)",
            "ValueErrorが発生する",
            f"ValueError: {expectedMessage}",
            testInvalidNumericValue,
        )

    print("\n種類別集計")
    for category, count in testCounts.items():
        print(f"{category}: {count}/{count}件合格")
    totalTestCount = sum(testCounts.values())
    print(f"全テスト合格: {totalTestCount}/{totalTestCount}")
