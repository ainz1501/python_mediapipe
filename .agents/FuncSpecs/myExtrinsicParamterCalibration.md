# myExtrinsicParameterCalibration 関数
8点アルゴリズムにより求めた相対的な外部パラメータに対し、DfDにより求めた深度情報をもとに2視点間の実空間上の外部パラメータに復元する
配置ファイル名: myCameraCalibration.py
使用言語: python


## 使用外部ライブラリ・モジュール
- NumPy: ランドマーク、カメラパラメータ、回転行列、並進ベクトルの表現に使用する
    - 呼び出し方: `import NumPy as np`
- eightPointAlgorithm: 独自モジュール。eightPointAlgorithm関数を呼び出す
    - 呼び出し方: `import eightPointAlgorithm as epa`
- depthFromDefocus: 独自モジュール。epthFromDefocus関数を呼び出す
    - 呼び出し方: `import depthFromDefocus as dfd`
 
## 使用独自クラス
### クラス名: FocalStack
記述ファイル名: `depthFromDefocus.py`
焦点スタックに含まれる異なる光学パラメータの値で撮影した各画像の画素値と撮影時の各光学パラメータ（焦点距離、物体側の合焦距離、レンズ開口径、ピクセルサイズ）をまとめたクラス

## 引数
|引数名|型|shape|要素の型|説明|
|---|---|---|---|---|
|landmark1|np.ndarray|(N, 2)|flort|カメラ1の画像から取得したN個の2Dランドマーク|
|landmark2|np.ndarray|(N, 2)|flort|カメラ2の画像から取得したN個の2Dランドマーク|
|focalStack|dfd.FocalStack|-|-|焦点スタックに含まれる複数枚の画像と光学パラメータ|
|intrinsicParameter|np.ndarray|(3, 3)|-|カメラの内部パラメータ行列|

## ハイパーパラメータ
記述箇所: 
|パラメータ名|型|デフォルト値|説明|
|---|---|---|---|

## 戻り値
|戻り値名|型|shape|要素の型|説明|
|---|---|---|---|---|
|rotationMatrix|np.ndarray|(3, 3)|float|カメラ間の回転行列|
|realScaleTranslation|np.ndarray|(3,)|float|実空間のスケールに復元したカメラ間の並進ベクトル|

## 処理フロー

1. `eightPointAlgorithm`関数に`landmark1`、`landmark2`、`intrinsicParameter`を渡し、`rotationMatrix`と大きさ1の並進ベクトルを取得する。
2. `depthFromDefocus`関数に`focalStack`を渡し、AIF画像と深度マップを取得する。
3. `recoverRealScaleTranslation`関数に`landmark1`、`landmark2`、深度マップ、`intrinsicParameter`、`rotationMatrix`、大きさ1の並進ベクトルを渡し、`realScaleTranslation`を取得する。
4. `rotationMatrix`と`realScaleTranslation`を、この順序で返す。

## 使用関数の呼び出し

### `eightPointAlgorithm`
記述ファイル名: eightPointAlgorithm.py

#### 入力

|引数名|型|shape|要素の型|説明|
|---|---|---|---|---|
|landmark1|np.ndarray|(N, 2)|float|カメラ1の画像から取得したN個の2Dランドマーク|
|landmark2|np.ndarray|(N, 2)|float|カメラ2の画像から取得したN個の2Dランドマーク|
|intrinsicParameter|np.ndarray|(3, 3)|float|カメラの内部パラメータ行列|

#### 出力

|戻り値名|型|shape|要素の型|説明|
|---|---|---|---|---|
|rotationMatrix|np.ndarray|(3, 3)|float|カメラ間の回転行列|
|unitTranslation|np.ndarray|(3,)|float|大きさを1に正規化したカメラ間の並進ベクトル|

```python
rotationMatrix, unitTranslation = epa.eightPointAlgorithm(
    landmark1,
    landmark2,
    intrinsicParameter,
)
```

戻り値は、回転行列、大きさ1の並進ベクトルの順とする。

### `depthFromDefocus`
記述ファイル名: depthFromDefocus.py

#### 入力

|引数名|型|shape|要素の型|説明|
|---|---|---|---|---|
|focalStack|dfd.FocalStack|-|-|焦点スタックに含まれる複数枚の画像と光学パラメータ|

#### 出力

|戻り値名|型|shape|要素の型|説明|
|---|---|---|---|---|
|aifImage|np.ndarray|(H×W, C)|float|全画素に焦点が合ったAIF画像を画素ごとに並べた行列|
|depthMap|np.ndarray|(H, W)|float|各画素に対応するカメラ中心からの深度|

```python
aifImage, depthMap = dfd.depthFromDefocus(focalStack)
```

戻り値は、AIF画像、深度マップの順とする。

### `recoverRealScaleTranslation`
記述ファイル名: myCameraCalibration.py

#### 入力

|引数名|型|shape|要素の型|説明|
|---|---|---|---|---|
|landmark1|np.ndarray|(N, 2)|float|カメラ1の画像から取得したN個の2Dランドマーク|
|landmark2|np.ndarray|(N, 2)|float|カメラ2の画像から取得したN個の2Dランドマーク|
|depthMap|np.ndarray|(H, W)|float|各画素に対応するカメラ中心からの深度|
|intrinsicParameter|np.ndarray|(3, 3)|float|カメラの内部パラメータ行列|
|rotationMatrix|np.ndarray|(3, 3)|float|カメラ間の回転行列|
|unitTranslation|np.ndarray|(3,)|float|大きさを1に正規化したカメラ間の並進ベクトル|

#### 出力

|戻り値名|型|shape|要素の型|説明|
|---|---|---|---|---|
|realScaleTranslation|np.ndarray|(3,)|float|実空間のスケールに復元したカメラ間の並進ベクトル|

```python
realScaleTranslation = recoverRealScaleTranslation(
    landmark1,
    landmark2,
    depthMap,
    intrinsicParameter,
    rotationMatrix,
    unitTranslation,
)
```

戻り値は、実空間のスケールに復元した並進ベクトルとする。

## 前提条件・例外

- `landmark1`と`landmark2`のランドマーク数Nは等しく、同じ行に同一地点の対応点が格納されている。
- 8点アルゴリズムを適用するため、ランドマーク数Nは8以上とする。
- 焦点スタックに含まれる画像の枚数は2枚以上
