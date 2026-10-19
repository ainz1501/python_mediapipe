# 目的
モジュール`depthFromDefocus.py`内に`optimizeAIFImage()`関数を作成する <br>
以下にその関数とその関数内で使用する各関数の内容を記述する

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
    │       │   ├── makeGaussianBlurKernel (depthFromDefocus)
    │       │   ├── calcLipschitz (depthFromDefocus)
    │       │   └── FISTA (depthFromDefocus)
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
##### makeGaussianBlurKernel (depthFromDefocus)
##### calcLipschitz (depthFromDefocus)
##### FISTA (depthFromDefocus)
#### bFinishUpdate (depthFromDefocus)
## recoverRealScaleTranslation (triangulate3DHPE)
-->

# optimizeAIFImage 関数
撮影した焦点スタック $J$ と深度マップから生成したガウスぼかしカーネル $A$ を用いて、AIF画像 $I$ を更新する。<br>
AIF画像の最適化にはFISTAによる $||AI-J||^2$ の巨大ベクトルの高速な最小化を行う。<br>
ステップサイズは $||AI-J||^2$ のリプシッツ定数の逆数となり、それを求めるために必要な $A$ の最大固有値はべき乗法とレイリー商による近似で求める。<br>
よって、ガウスぼかしカーネルの生成と焦点スタックの巨大な行列を作成し、リプシッツ定数を求めたのち、FISTAによる最適化を行う。

## 入力・出力
（*は関数の出力を示す）
- 深度マップ $\R^{H \times W}$
- 焦点スタック $K$ 枚
    - 画像 $\R^{WH \times C}$
    - 撮影時のパラメータ 
- AIF画像 $\R^{WH \times C}$
- AIF画像* $\R^{WH \times C}$
- ガウスぼかしカーネル行列* $\R^{KWH \times WH}$
- 焦点スタック行列* $\R^{KWH \times C}$

$W,H$ はそれぞれ使用画像の幅と高さ（入力画像は同じ画像サイズ）であり、 $C$ はチャネル数である（RGB画像なら、3チャネル）。そして、 $K$ は焦点スタックに含まれる画像の枚数を示している

## 使用するハイパーパラメータ（モジュール内で定義するパラメータ）
- 最大カーネルサイズ
- べき乗法の反復回数
- FISTAの初期反復回数
- FISTAの反復回数増加倍率

## 処理フロー
- ガウスぼかしカーネルの作成
    - 関数名: makeGaussianBlurKernel
    - 入力・出力
        - 深度マップ `depthMap`
        - 焦点スタック `focalStacks`
            - 撮影した各画像 `focalStacks[].img`
            - 撮影時のカメラのパラメータ `focalStacks[].params`
        - 最大カーネルサイズ `maxKernelSize`
        - ガウスぼかしカーネル* `A`
            - 焦点スタックの各画像のガウスぼかしカーネルを配列に収めた巨大な疎行列
            - $A\in \R^{(K \cdot WH) \times WM }$
- 焦点スタックの巨大なベクトル `J` の作成
    - 画像を列ベクトル状にまとめた `focalStacks[i].img` を取得し、`J` の列方向にそのまま追加する
        - 最終的に $ J \in \R^{(KWH \times C)}$ の形となる
- リプシッツ定数を求める
    - 関数名: calcLipschitz
    - 入力・出力
        - ガウスぼかしカーネル `A`
        - べき乗法の反復回数 `powerIterationTimes`
        - リプシッツ定数 `L`*
- FISTAによる最適化
    - 関数名: FISTA
    - 入力・出力
        - 焦点スタックベクトル `J`
        - 更新前AIF画像ベクトル `I`
        - ガウスぼかしカーネル `A`
        - FISTAのパラメータ
            - FISTAの初期反復回数 `FISTAIterationTimes`
            - FISTAの反復回数増加倍率 `FISTAiterationMagnifications`
- 最適化した新たな [AIF画像] を出力する
        

## makeGaussianBlurKernel関数
推定した深度マップから焦点スタックの各画像に対応する巨大な疎行列 `A` を計算する。<br>
`A` とはAIF画像にかける各ガウスぼかしカーネルで、対応する焦点スタックの画像のぼかしカーネルをまとめた配列である。そして、 $J_k \in \R^{mn}, I \in \R^{mn}$ のようにベクトルとするとき、$A_k \in \R^{mn \times mn}$ の巨大な行列であり、焦点スタックの1画像の1画素に対応するぼかしカーネルは `A` の1行分となる。ぼかしカーネルのサイズよりも外側の部分は全て0であるため `A` は疎行列となる

### 入力・出力
（*は関数の出力を示す）
- 深度マップ `depthMap`
    - 画像サイズ `[W, H]` と等しいサイズ
- 焦点スタック `J`
    - 撮影時のカメラのパラメータ `J[].params`
- 最大カーネルサイズ `maxKernelSize`
- ガウスぼかしカーネル* `A`
    - 焦点スタックの各画像のガウスぼかしカーネルを配列に収めた $K$ 個の巨大な疎行列の配列
        - $A \in \R^{K \times W \times H}$ 

### 処理フロー
- カーネルの半径を計算 `r = maxKernelSize/2`
- 焦点スタックの画像ごとにループ `J[i]`
    - 各画素ごとにループ `[x,y]`
        - ボケ半径を計算する `blurRadius = calcBlurRadius(depthMap[y][x], J[i].params)`
        - ガウスぼかしカーネルを計算 `blurKernel = GaussianPSF(depthMap[y][x])`
        - ガウスぼかしカーネルを正規化 `blurKernel = blurKernel/sum(blurKernel)`
        - 0で初期化した巨大な行ベクトル `ARow` $(\R^{width\cdot height})$ を作成
        - `rx, ry` で $[-r,r]$ の範囲で2重ループ 
            - `ARow[(y+ry)*W+(x+rx)] = blurKernel[ry][rx]` と更新する
        - `A` の新たな行として追加
- 全ループ終了後、`A` を返す

## calcLipschitz関数
FISTAのステップサイズを求めるためのリプシッツ定数を求める関数。<br>
リプシッツ定数にはガウスぼかしカーネルの行列 `A` の最大固有値が式に含まれる。そのため、まずはべき乗法を用いて、全要素を1としたベクトルを正規化した `v` を `A` の最大固有値に対応する固有ベクトルに近似していく。そして、近似したベクトルを用いて、レイリー商から最大固有値を求める。<br>
最後に、求めた固有値を用いて、リプシッツ定数を求めていく。<br>

### 入力・出力
（*は関数の出力を示す）
- ガウスぼかしカーネル行列 `A`
- べき乗法の反復回数 `powerIterationTimes`
- リプシッツ定数 `L`*

### 処理フロー
- 画素数を取得する（行列 `A` の１行のサイズ $mn$ ）
- 近似を行うベクトルを初期化する
    - 全ての要素を1とする $\R^{mn}$ のベクトル `v` を定義する
    - ベクトル `v` を正規化 `v = v / sum(v)`
- `powerIterationTimes` 回 `A` を適応する `v = A @ v`
- レイリー商に基づいて最大固有値を求める `maxEigenvalue = v.T @ A.T @ A @ v`

# 入力
- 焦点スタック $J_k$ 
    - 異なる焦点距離で撮影した各画像
        - 撮影した画像の枚数を $K$ とする
    - 画像サイズを $m\times n$ とすると、 $mn \times 1$ の巨大なベクトル 
- 撮影時のカメラのパラメータ
    - 焦点距離 $f$
    - 開口径 $D$
    - 物体側の合焦距離 $Z_f$
- 初期AIF画像 $I_{init}$
    - 全ての画素でピントが合っている (AIF) 画像の初期値
    - 画像サイズを $m \times n$ とすると、 $mn \times 1$ の巨大なベクトル
- 想定される深度範囲 $near, far$ 
- グリッドサーチでのパラメータ
    - 深度サンプリング数 $n$
- 黄金分割探索でのパラメータ
    - 最大反復回数
    - 許容誤差
- FISTAのパラメータ
    - 初期反復回数 $ T_0 $ 
    - 反復回数増加倍率 $\alpha$



# 理論
- 深度マップが固定された時
- べき乗法により $A$ の最大固有値を近似する
    - $v^{(k)}=A^TAv^{(k−1)} $
    - $λ_{max}≈\frac{(v^{(k)})^Tv^{(k)}}{(v^{(k)})^Tv^{(k−1)}}$
- $ y_1 = x_0 = I_{init}, t_1 = 1$ として初期化
- 以下の操作を繰り返す $(k\ge 1)$ 
    - $x_k = p_L(y_k)$ を計算
        - $p_L(y) = \text{argmin}_x \left\{ g(x) + \frac{L}{2} \left\| x - \left( y - \frac{1}{L} \nabla f(y) \right) \right\|^2 \right\}$
            - $f(y) = ||Ay-J||^2$
            - $g(x) = |x|_1$ 
    - $t_{k+1}=\frac{1+\sqrt{1+4t_k^2}}{2}$ により加速係数を計算
    - $y_{k+1}​=x_k​+\left( \frac{t_k​−1}{t_{k+1}​}​\right)(x_k​−x_{k−1}​)$ により次の補助点を計算
    - 反復回数は $\alpha^i T_0 $ 回行う
        - $i$ は交互最小化を周回した回数
