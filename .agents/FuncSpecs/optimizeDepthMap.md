# 目的
モジュール`depthFromDefocus.py`内に`optimizeDepthMap()`関数を作成する

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
    │       │   ├── gridSearch (depthFromDefocus)
    │       │   └── goldenSectionSearch (depthFromdefocus)
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
##### gridSearch (depthFromDefocus)
##### goldenSectionSearch (depthFromdefocus)
#### optimizeAIFImage (depthFromDefocus)
#### bFinishUpdate (depthFromDefocus)
## recoverRealScaleTranslation (triangulate3DHPE)
 -->

# optimizeDepthMap 関数
初期AIF画像または更新されたAIF画像にガウスぼかしカーネルをかけて作成したぼけスタック（blur stacks）と入力された焦点スタック（focal stacks）とでMSEを計算し、画素ごとに並列で深度を最適化する<br>
深度が決まれば順モデル（forward model）からぼかしカーネルが求められることから、グリッドサーチと黄金分割探索を用いて画素ごとに最適化を行う。

## 入力・出力
（*は関数の出力を示す）
- AIF画像 $I$（最初のループでは初期AIF画像 $I_{init}$ を入力）
- 焦点スタック $J$
- 撮影時のカメラパラメータ
- 深度マップ $Z$ *

## ハイパーパラメータ
以下のパラメータは`depthFromDefocus.py`モジュール内部で設定する<br>
引数には含めない

- 最大カーネルサイズ `maxKernelSize`
    - ガウスぼかしをかけるある画素を中心として、このサイズより外側の画素にかけるガウスぼかしを全て0とする
- グリッドサーチでのパラメータ
    - 想定される深度範囲
        - 写っている物体がカメラから見てどの深度にありうるかを設定する値
        - 最小深度 `near` 、最大深度 `far` の2つ
            - 単位はm
    - 深度サンプリング数 `SamplingNum`
        - 深度範囲の中から等間隔に取得するサンプリングの数を決定する
- 黄金分割探索でのパラメータ
    - 最大反復回数 `maxIteretionLim`（本誌では100回）
    - 許容誤差 `comvergenceTolerance`（本誌では $10^{-2}$ ）


## 処理フロー
- 以下を画素ごとに行う（並列可）
    - 0で初期化したの深度マップを定義
    - グリッドサーチによる大まかな深度推定
        - 関数名: gridSearch
        - 入力・出力
            - AIF画像
            - 焦点スタック
                - 撮影時のカメラのパラメータ
            - 想定される深度範囲
            - 深度サンプリング数
            - 最適な深度サンプルの前後のサンプルの深度*
                - 最適近傍深度*
                - 最適遠方深度*
    - 黄金分割探索による詳細な深度推定
        - 関数名: goldenSectionSearch
        - 入力・出力
            - 焦点スタック
                - 撮影時のカメラのパラメータ
            - 最適な深度サンプルの前後の深度
            - 最大反復回数
            - 許容誤差
            - 1画素の深度*
    - 対応する位置の深度マップに値を代入する
- 深度マップ* を返す

## gridSearch関数
入力された深度範囲から等間隔に取得した深度サンプルのうち、最も焦点スタックに近い画像が生成された深度を求め、その前後のサンプルの深度を返す<br>
求められる深度は1画素の深度であることに注意

### 入力・出力
（*は関数の出力を示す）
- AIF画像 `I`
- 焦点スタック `J`
    - 焦点スタックの各画像用の独自クラス `FocalStack` を要素にもつ配列
        - 画像 `J[].img` 
        - 撮影時のカメラのパラメータ `J[].params`
            - 焦点距離 `.f`
            - レンズ開口径 `.D`
            - 物体の合焦距離 `.Zf`
            - ピクセルサイズ `.ps`
- 最大カーネルサイズ `maxKernelSize`
- 想定される深度範囲 `near`, `far`
- 深度サンプリング数 `n`
- 最適な深度サンプルの前後のサンプルの深度*
    - 最適近傍深度 `optNear`
    - 最適遠方深度 `optFar`

### 処理フロー
- `far` と `near` の間の深度から、等間隔に `n` 個の深度サンプル配列 `depthSamples` を取得する
    - `depthSamples` は深度が昇順になるようになっている
- 各深度サンプル `depthSamples[i]` に対して焦点スタックの各画像のパラメータ `J[j].params` を用いてボケ半径 `blurRadius[j]`を計算
    - $b=D\frac{|Z-Z_f|}{Z}\frac{f}{Z_f-f}$, $ blurRadius = \frac{b}{2ps}$ 
        - $Z$ が `depthSamples[i]` に対応
- 以下の処理で各焦点スタックに対応するガウスぼかしカーネル `blurKernel[j]` を生成する
    - $\R^{2r \times 2r}$ の、`A[j]=GaussianPSF(blurRadius[j])` を求める
        - `r` は最大カーネルサイズの半径であり、`r=maxKernelSize/2` となる
        - $G_{i,j}(u,v)=\exp(-\frac{u^2+v^2}{2\sigma^2})$ となり、これを計算する関数が `GaussianPSF()`
- ガウスぼかしカーネルを正規化する $\bar{G}_{i,j}(u,v)=\frac{G_{i,j}}{\sum_{u=-r}^r\sum_{v=-r}^r G_{i,j}(u,v)}$
- 作成したガウスぼかしカーネルをAIF画像に畳み込んだぼけスタック `B[j]` を作成する
    - RGBチャネルがあるため、 $ \R^{j\times 3}$ となる二次元配列
- 以下の数式に従って、MSEを計算する
    - $e(i,j)=\frac{1}{KC}\sum^K_{k=1} \sum^C_{c=1} \left(J_k(i,j,c)-\hat{J}_k(i,j,c)\right)^2$
    - 計算結果は各深度サンプルに対応する `MSEResult[i]` に格納しておく
- 最小のMSEを持つ深度を特定し、そのインデックスを `optMSEIdx` に代入
- その前後の深度サンプルを出力する
    - `optNear = depthSamples[optMSEIdx-1]`
    - `optFar = depthSamples[optMSEIdx+1]`

## goldSectionSearch関数
求めた最適な深度より細かい詳細な深度を求めるため、黄金分割探索を行うことで求めていく。<br>
黄金分割探索は、提示された領域で最適な箇所が必ず存在するという前提（凸関数の最小化）で、その領域を黄金比で分けたのち、どちらの領域に最適解が存在するのかを選択する。<br>
これを目的のMSE値になるか、指定した反復回数を繰り返すことで、より詳細な範囲で最適な深度を求めることができる。

### 入力・出力
（*は関数の出力を示す）
- 最適な深度サンプルの前後の深度
    - `optNear`
    - `optFar`
- 焦点スタック
- 最大反復回数 `maxIteretionLim`
- 許容誤差 `comvergenceTolerance`
- 1画素の深度* 

### 処理フロー
- グリッドサーチで求めた `optNear` から `optFar` までの範囲で黄金比 $1: \phi$ になるような `sectionDepth` を決める 
    - $\phi = \frac{1+\sqrt{5}}{2} \approx 1.618 $
    - `sectionDepth = (phi*optFar + optNear)/(1+phi)`
- その3点の中で広い方の区間内で新たに `nextSectionDepth` を決める
    - `nextSectionDepth = a + (b - a)` となる
- `sectionDepth, nextSectionDepth` それぞれでMSEを計算
    - `MSESectionDepth, MSENextSectionDepth` にそれぞれ代入
- `MSENextSectionDepth < MSESectionDepth` の場合
    - 探索区間を `optNear = sectionDepth, optFar = optFar` に設定し直す
- `MSENextSectionDepth >= MSESectionDepth` の場合
    - 探索区間を `optNear = optNear, optFar = nextSction` に設定し直す
- 反復回数が `maxIteretionLim` に達するか、 `min(MSENextSectionDepth, MSESectionDepth) <= comvergenceTolerance` を満たすまで上記の操作を繰り返す
    - 条件に達した場合、深度マップの1画素として `min(MSENextSectionDepth, MSESectionDepth)` を返す






