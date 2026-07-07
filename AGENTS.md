# triangulate3DHPE.py 改善方針

## 目的

`triangulate3DHPE.py` を、スクリプトとしてだけでなく Python モジュールとしても使いやすくする。

現在は import 時にキャリブレーションファイルを読み込むなど、外部ファイルや固定パスへの依存が強い。今後は、必要な設定やパスを関数引数で渡せる構造に整理する。

## 変更方針

- import しただけでファイル読み込みや重い処理が走らないようにする。
- 実行処理は `if __name__ == "__main__":` 配下、または明示的な `run_*` 関数に置く。
- データセット名、カメラ番号、入力パス、出力パス、フレーム範囲などはグローバル固定値だけに依存させず、関数引数で指定できるようにする。
- 既存コードから呼ばれている関数名はできるだけ維持する。
  - `Landmark_detect`
  - `Normalized_to_screen_coord`
  - `Projection_mat_calc`
  - `Triangulate_3Dpoint`
  - `Triangulate3DHPE`
- 新しい関数を追加する場合は、既存関数を壊さずラッパーとして追加する。
- 返り値は後続処理で使いやすいように、可能な限り `numpy.ndarray`、`dict`、`list[dict]` のような明確な型にする。
- デバッグ用 `print` は必要最小限にし、将来的には `verbose` 引数や logging に置き換える。

## 注意点

- `calcurateCamPose.py` は `triangulate3DHPE` の `K_left`, `R_left`, `T_left`, `K_right`, `R_right`, `T_right` を直接参照しているため、グローバル変数を削除する場合は互換用 API を用意する。
- `triangulate3DHPE_show_result.py` は `Landmark_detect` を直接利用しているため、関数シグネチャを変更する場合は呼び出し側も更新する。
- Panoptic Dataset のディレクトリ構造に依存している処理は、パス生成関数や設定オブジェクトに分離する。
- 画像・動画・ランドマーク JSON の出力先はハードコードせず、引数で指定可能にする。

## 確認方法

- `python triangulate3DHPE.py` で従来の実行ができること。
- `import triangulate3DHPE` だけでファイル読み込みエラーが起きないこと。
- `run_triangulate3dhpe_data(...)` に任意のパスやフレーム範囲を渡して実行できること。
- 既存の `triangulate3DHPE_show_result.py` から `Landmark_detect` が使えること。
- 出力 JSON の形式を維持すること。

## 依存関係

- 主な依存ライブラリは `opencv-python`, `mediapipe`, `numpy`, `matplotlib`。
- MediaPipe の検出結果オブジェクトは独自型なので、保存・テスト用には必要に応じて `numpy.ndarray` や `dict` に変換する。
- OpenCV の画像は BGR、MediaPipe 入力は RGB である点に注意する。

## 入出力仕様

- 入力画像は OpenCV で読み込んだ BGR 画像を想定する。
- 2D ランドマークは画像座標系の `(x, y)` 配列として扱う。
- 3D ランドマークは `(N, 3)` の `numpy.ndarray` として扱う。
- 保存する推定結果 JSON は既存形式を維持する。
  - `{"frame": frame_num, "landmarks": {"0": [x, y, z], ...}}`
  - 人物が検出できない場合は `"landmarks": null`

## 座標系と単位

- MediaPipe の正規化座標は画像幅・高さを使ってスクリーン座標に変換する。
- `cv2.triangulatePoints` に渡す 2D 点は左右画像で対応順が一致している必要がある。
- Panoptic Dataset の GT 座標と推定 3D 座標の単位・座標系が一致しているか、MPJPE 計算前に確認する。

## エラーハンドリング

- 入力動画、画像、キャリブレーション JSON が存在しない場合は、分かりやすい例外を出す。
- MediaPipe が人物を検出できない場合は例外ではなく、空配列または `None` を返す。
- 出力ディレクトリは保存前に `os.makedirs(..., exist_ok=True)` で作成する。

## テスト方針

- 実データセット全体に依存しない小さいテストを優先する。
- 可能なら `testIO` 配下の小さい入力・出力で動作確認できるようにする。
- import 時に副作用がないことを確認するテストを追加する。
- `Projection_mat_calc`, `Triangulate_3Dpoint`, `compute_mpjpe` は小さい配列で単体確認しやすい形を保つ。

## 命名と互換性

- 既存の CamelCase 関数名は互換性のため残す。
- 新規追加する内部関数は可能なら CamelCase にする。
- 既存関数を置き換える場合は、古い関数をラッパーとして残して段階的に移行する。

# 8点アルゴリズムによる外部パラメータ推定

## 目的

左右画像の対応する2D点から、8点アルゴリズムを用いて基礎行列または本質行列を推定し、カメラ間の相対姿勢 `(R, t)` を求めるプログラムを作成する。

この処理は `triangulate3DHPE.py` の三角測量処理と組み合わせて使う可能性があるが、まずは独立したモジュールとして実装する。

## 実装方針

- 8点アルゴリズム本体は、画像読み込みや MediaPipe 検出に依存しない純粋な数値計算関数として実装する。
- 入力は左右画像の対応点配列を基本とする。
  - `points_left`: shape `(N, 2)` の `numpy.ndarray`
  - `points_right`: shape `(N, 2)` の `numpy.ndarray`
  - `N >= 8`
- 対応点の順序は左右で一致している必要がある。
- 画像座標から基礎行列 `F` を求める関数と、カメラ内部パラメータ `K_left`, `K_right` を使って本質行列 `E` を求める関数を分ける。
<!--
- 数値安定性のため、正規化8点アルゴリズムを基本とする。
  - 各画像の2D点を重心0、分散を1になるように正規化する。
  - 正規化後の点で線形方程式 `A f = 0` を作る。
  - `np.linalg.svd(A)` の最小特異値に対応する右特異ベクトルから `F` を得る。
  - `F` は rank 2 制約を満たすように、SVD 後に最小特異値を0にして再構成する。
  - 最後に正規化変換を戻す。
-->
- 本質行列 `E` は `E = K_right.T @ F @ K_left` で求める。
<!--
- 本質行列にも制約をかける。
  - SVD で `E = U @ diag(s) @ Vt`
  - 特異値を `(1, 1, 0)` または `((s0+s1)/2, (s0+s1)/2, 0)` に補正する。
-->
- `R, t` の復元は `E` の SVD から行う。
  - 候補は4通りあるため、三角測量後に正の奥行きになる点が最も多い候補を選ぶ。
  - `t` はスケール不定なので、単位ベクトルまたは任意スケールの並進方向として扱う。

## 推奨する関数構成

<!--
- `Normalize_points(points)`  
  2D点を正規化し、正規化点と変換行列 `T` を返す。

- `Estimate_fundamental_matrix(points_left, points_right)`  
  正規化8点アルゴリズムで基礎行列 `F` を推定する。
-->

- `Estimate_fundamental_matrix(points_left, points_right)`  
  非正規化の8点アルゴリズムで基礎行列 `F` を推定する。

- `Estimate_essential_matrix(points_left, points_right, K_left, K_right)`  
  `F` と内部パラメータから本質行列 `E` を推定する。

- `Decompose_essential_matrix(E)`  
  本質行列から `(R, t)` 候補を4通り生成する。

- `Choose_correct_pose(pose_candidates, points_left, points_right, K_left, K_right)`  
  三角測量して正の奥行き条件を満たす候補を選ぶ。

- `Estimate_relative_pose_eight_point(points_left, points_right, K_left, K_right)`  
  対応点から最終的な相対姿勢 `(R, t, E, F)` を返す高レベル関数。

## 入出力仕様

- 入力点は OpenCV 画像座標系の `(x, y)` とする。
- 内部パラメータ `K_left`, `K_right` は shape `(3, 3)` の `numpy.ndarray` とする。
- `F` と `E` は shape `(3, 3)` の `numpy.ndarray` とする。
- `R` は shape `(3, 3)` の回転行列とする。
- `t` は shape `(3,)` または `(3, 1)` の並進方向ベクトルとする。
- 戻り値は後続処理で使いやすいように、必要に応じて `dict` でまとめる。
  - 例: `{"R": R, "t": t, "E": E, "F": F, "inlier_mask": mask}`

## エラーハンドリング

- 対応点が8点未満の場合は `ValueError` を出す。
- 左右の対応点数が一致しない場合は `ValueError` を出す。
- 入力点が shape `(N, 2)` でない場合は `ValueError` を出す。
- SVD の結果が退化している場合は、分かりやすい例外または `status` を返す。
- 外れ値を含む実データでは RANSAC が必要になるため、最初の純粋な8点実装とは別に RANSAC 付き関数を追加する。

## OpenCV との関係

- 自作実装を主とし、検証用に `cv2.findFundamentalMat`, `cv2.findEssentialMat`, `cv2.recoverPose` と結果を比較してよい。
- OpenCV 関数だけに置き換えない。8点アルゴリズムの学習・検証が目的であるため、SVD による計算過程をコード上で確認できるようにする。

## テスト方針

- まずは合成データでテストする。
  - 既知の `R, t, K_left, K_right` を作る。
  - ランダムな3D点を左右画像へ投影する。
  - 投影点から `Estimate_relative_pose_eight_point` を実行する。
  - 復元した `R` と `t` の向きが既知値に近いことを確認する。
- ノイズなしデータで通ることを確認してから、少量のガウスノイズを加えて確認する。
- 実画像や MediaPipe ランドマークを使うテストは後段にする。
- `t` はスケール不定なので、長さではなく方向の一致を評価する。
- `R` は `R.T @ R ≈ I`、`det(R) ≈ 1` を確認する。

## triangulate3DHPE.py との連携

- `triangulate3DHPE.py` の `Projection_mat_calc` や `Triangulate_3Dpoint` は再利用してよい。
- ただし、8点アルゴリズム用モジュールは `triangulate3DHPE.py` を import しただけで重い処理が走らない前提を守る。
- 既存の Panoptic キャリブレーションから得た外部パラメータと、8点アルゴリズムで推定した外部パラメータを比較できるようにする。

## 初期テスト方針: 非正規化・補正なし8点アルゴリズム

まずはアルゴリズムの流れを確認するため、正規化も特異値補正も行わない最小実装を作る。

この段階では精度や数値安定性よりも、次の処理手順をコードで追えることを優先する。

1. 左右画像の対応点 `points_left`, `points_right` を受け取る。
2. 画像座標をそのまま使って線形方程式 `A f = 0` を作る。
3. `np.linalg.svd(A)` を実行する。
4. 最小特異値に対応する右特異ベクトル `Vt[-1]` を `3x3` に reshape して `F` とする。
5. `E = K_right.T @ F @ K_left` で本質行列を作る。
6. `E` をそのまま SVD 分解し、4通りの `(R, t)` 候補を作る。
7. 三角測量して、正の奥行きになる点が最も多い候補を選ぶ。

### この段階で行わないこと

- 2D点の正規化は行わない。
- `F` の rank 2 補正は行わない。
- `E` の特異値を `(s, s, 0)` にする補正は行わない。
- RANSAC による外れ値除去は行わない。

### 推奨する関数構成

- `Estimate_fundamental_matrix_raw(points_left, points_right)`  
  非正規化・rank補正なしで `F` を求める。

- `Estimate_essential_matrix_raw(points_left, points_right, K_left, K_right)`  
  `F` から `E = K_right.T @ F @ K_left` を求める。`E` の特異値補正は行わない。

- `Decompose_essential_matrix_raw(E)`  
  未補正の `E` から `(R, t)` 候補を作る。

- `Estimate_relative_pose_eight_point_raw(points_left, points_right, K_left, K_right)`  
  非正規化・補正なしの結果を `dict` で返す。

### 注意点

- 未補正の `F` は一般に rank 3 になり得る。
- 未補正の `E` は理想的な本質行列の条件 `singular values = (s, s, 0)` を満たさないことがある。
- そのため、復元された `R` が完全な回転行列からずれる可能性がある。
- この実装は比較・学習用の基準線とし、後で正規化あり・補正あり実装と比較する。

### テスト方針

- まずノイズなし合成データで動作を確認する。
- `F` の rank、`E` の特異値、`det(R)`、`R.T @ R` を出力して、補正なしでどれくらい崩れるか確認する。
- 正規化あり・補正あり版と同じ入力で比較できるように、戻り値のキー名はできるだけ揃える。
