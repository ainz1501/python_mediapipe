# alternatingMinimization  関数
各画素の深度をまとめた深度マップと全ての画素が合焦しているAIF画像を交互に最適化していく

配置ファイル名: depthFromDefocus.py
使用言語: python

## 使用外部ライブラリ・モジュール
- NumPy: ランドマーク、カメラパラメータ、回転行列、並進ベクトルの表現に使用する
    - 呼び出し方: `import NumPy as np`

## 使用独自クラス
- FocalStack: 焦点スタックを保存する変数に使用

## 引数
|引数名|型|shape|要素の型|説明|
|---|---|---|---|---|
|initialAIFImage|np.array|(HW,C)|float|`initAIFImage`により作成したAIF画像|
|focalStack|dfd.FocalStack|-|-|焦点スタックに含まれる複数枚の画像と光学パラメータ|

## ハイパーパラメータ
記述箇所: 
|パラメータ名|型|デフォルト値|説明|
|---|---|---|---|
|goalMSE|float|0.01|反復の終了条件として使用する平均二乗誤差距離|
|epoch|int|40|交互最小化の反復回数の上限|

## 戻り値
|戻り値名|型|shape|要素の型|説明|
|---|---|---|---|---|
|depthMap|np.array|(H, W)|float|各シーンの画素におけるカメラ中心（レンズ）からの深度情報|
|AIFImage|np.array|(HW,C)|float|全画素が合焦しているAIF (All-In-Focus) 画像|

## 処理フロー
1. `AIFImage`に`initalAIFImage`を代入
2. `J`に`focalStack.allImg`を代入
2. 以下を`epoch`回繰り返す
3. `optimizeDepthMap`関数に`AIFImage`と`focalStack`を入力し、`depthMap`を返す
4. `optimizeAIFImage`関数に`depthMap`、`focalStack`、`AIFImage`を入力し、`gaussianFullKernel`と`AIFImage`を返す
5. `blurStack`に`A @ I`

## 使用関数の呼び出し
### `関数名`
記述ファイル名: 

#### 入力

|引数名|型|shape|要素の型|説明|
|---|---|---|---|---|

#### 出力

|戻り値名|型|shape|要素の型|説明|
|---|---|---|---|---|

```python
実際の呼び出しコード

```

戻り値の順序や内容を一言

## テスト方法
同ファイル上に以下の内容のテストコードを記述する
- テストコードの記述箇所: 
- `if __name__ == '__main__':` の下にテストコードを記述する
- 引数となるデータの型を変化させ、適切でないデータ型が入力される全パターンを記述する

## 前提条件・例外


---- [参考内容] ----
# 目的
モジュールdepthFromDefocus.py内に、AIF画像と深度マップの最適化を交互に行う`alternatingMinimization`関数を作成する

# 関数のツリー構造図
() はその関数を定義しているモジュール名を示す
```
.
└── myExtrinsicParameterCalibration (triangulate3DHPE)
    ├── eightPointAlgorithm (eightPointAlgorithm)
    ├── depthFromDefocus (depthFromDefocus)
    │   ├── initAIFImage (depthFromDefocus)
    │   └── alternatingMinimization (depthFromDefocus)
    │       ├── optimizeDepthMap (depthFromDefocus)
    │       ├── optimizeAIFImage (depthFromDefocus)
    │       └── bFinishUpdate (depthFromDefocus)
    └── recoverRealScaleTranslation (triangulate3DHPE)
```
<!-- 構造図一時保存用
# myExtrinsicParameterCalibration (triangulate3DHPE)
## eightPointAlgorithm (eightPointAlgorithm)
## depthFromDefocus (depthFromDefocus)
### initAIFImage (depthFromDefocus)
### alternatingMinimization (depthFromDefocus)
#### optimizeDepthMap (depthFromDefocus)
#### optimizeAIFImage (depthFromDefocus)
#### bFinishUpdate (depthFromDefocus)
## recoverRealScaleTranslation (triangulate3DHPE)
 -->

# alternatingMinimization関数
初期AIF画像をもとに、深度マップとAIF画像を交互に最適化していく

## 入力・出力
- 初期AIF画像
- 焦点スタック
    - 撮影時のパラメータが異なる画像
    - 撮影時のカメラのパラメータ
- 深度マップ*
- AIF画像*

## ハイパーパラメータ （depthFromDefocusモジュール内で定義）
以下のパラメータはモジュール内部で設定する。引数には含めない
- 反復の終了条件パラメータ
    - 目標MSE値
    - エポック数（反復回数の上限）
- グリッドサーチでのパラメータ
    - 想定される深度範囲
        - 写っている物体がカメラから見てどの深度にありうるかを設定する値
        - 最小深度、最大深度の2つ
            - 単位はm
    - 深度サンプリング数
        - 深度範囲の中から等間隔に取得するサンプリングの数を決定する
- 黄金分割探索でのパラメータ
    - 最大反復回数（本誌では100回）
    - 許容誤差（本誌では $10^{-2}$ ）
- FISTAのパラメータ
    - べき乗法の反復回数（今回は100回と設定）
    - 初期反復回数 $ T_0 $ 
    - 反復回数増加倍率 $\alpha$

## 処理フロー
（*は関数の出力を示す）
- 以下の処理をループ
    - 深度マップの最適化
        - 関数名: optimizeDepthMap
        - 入力・出力
            - AIF画像（最初のループでは初期AIF画像を入力）
            - 焦点スタック
            - グリッドサーチでのパラメータ
            - 黄金分割探索でのパラメータ
            - 深度マップ*
    - AIF画像の最適化
        - 関数名: optimizeAIFImage
        - 入力・出力
            - 深度マップ
            - 焦点スタック
            - AIF画像
            - FISTAのパラメータ
            - AIF画像*
            - ガウスぼかしカーネル行列*
                現在の深度マップから求めた各画素に適応するガウスぼかしカーネルをまとめた巨大な疎行列
            - 焦点スタック行列*
    - ループ終了判定
        - 関数: bFinishUpdate
        - 入力・出力
            - 反復の終了条件パラメータ
            - AIF画像
            - 焦点スタック行列
            - ガウスぼかしカーネル行列
            - 反復終了判定ブール値* `flag`
    - もし、`flag = true` なら `break` してループを脱出
- ループ終了後、深度マップ* とAIF画像* を返す

## optimizeDepthMap関数
初期AIF画像または更新されたAIF画像にガウスぼかしカーネルをかけて作成したぼけスタック（blur stacks）と入力された焦点スタック（focal stacks）とでMSEを計算し、画素ごとに並列で深度を最適化する
深度が決まれば順モデル（forward model）からぼかしカーネルが求められるため、グリッドサーチと黄金分割探索を用いて最適化可能
詳細は`optimizeDepthMap.md`を参照

## optimizeAIFImage関数
各画素のぼかしカーネルをまとめた巨大な素行列 $A$ に対し、AIF画像 $I$ をかけることで焦点スタック $J$ になるという線形な関係式から、FISTAを用いて $AI=J$ よりAIF画像を最適化する
詳細は`optimizeAIFImage.md`を参照

## bFinishUpdate関数
AIF画像と深度マップの最適化はそれぞれ交互に反復しながら最適化される
この関数は終了条件を満たしているかを判定し、その判定に応じたブール値を返す関数

### 入力・出力
（*は関数の出力を示す）
- 反復の終了条件パラメータ
    - 目標MSE値 `MSEThreshold`
    - エポック数（反復回数の上限） `epoch`
- 現在のループ回数 `nowEpoch`
- AIF画像 `I`
- ガウスぼかしカーネル行列 `A`
- 焦点スタック行列 `J`
- 反復終了判定ブール値* `flag`

### 処理フロー
- もし `nowEpoch > epoch` なら `flag = true` として返す
- 各ぼけスタック `B = A @ I` を計算する
- 各画素の誤差をまとめた `errorMat = J - B` を計算
- 全画素、全チャネルの誤差を2乗した後合計し、 $1/KHWC$ で割った値を `MSE`（MSEを求める）
- もし、`MSE <= MSEThreshold` であれば、`flag = true` として返す
- 上記全てに当てはまらない場合は、`flag = false` として返す


