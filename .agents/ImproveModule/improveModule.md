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