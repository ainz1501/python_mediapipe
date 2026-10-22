# 更新履歴

## v5.2.2（2026-10-22）

### `src/modules/depthFromDefocus.py`

- FocalStackテストの入力表示を、引数名だけでなく型と具体的な値まで確認できる形式へ変更
- `np.ndarray`の入力について、shape、要素の型および全要素を表示する処理を追加
- 異常系テストでも、変更した不正値だけでなく、対象へ同時に渡す全引数を表示するよう変更
- 表示用入力と実際のテスト入力を共通の引数生成処理から作成し、両者が一致する構成へ変更

### 動作確認

- `.venv/bin/python`で`src/modules/depthFromDefocus.py`の構文チェックに合格
- `FocalStack`のimportチェックに合格
- FocalStackの個別テスト1件、統合テスト1件、異常系テスト9件の全11件に合格
- 正常系と異常系の全ケースで、実際に渡す入力の型と具体的な値が省略されず表示されることを確認

## v5.2.1（2026-10-21）

### `src/modules/depthFromDefocus.py`

- `TEST_TARGET`の値によって、同一ファイル内のFocalStackテストとDepth from Defocusテストの実行・表示対象を切り替える処理を追加
- `TEST_TARGET = "FocalStack"`を設定し、FocalStackに関するテストだけを実行・表示する構成へ変更
- テスト番号と種類別集計を、選択した対象の実行件数だけで採番・集計するよう変更
- Depth from Defocusのテストコードは同一ファイル内に保持し、対象外の場合は実行しない

### 動作確認

- `.venv/bin/python`で`src/modules/depthFromDefocus.py`の構文チェックに合格
- `FocalStack`のimportチェックに合格
- FocalStackの個別テスト1件、統合テスト1件、異常系テスト9件の全11件に合格
- ターミナルにFocalStackのテストと集計だけが表示されることを確認

## v5.2.0（2026-10-21）

### `src/modules/depthFromDefocus.py`

- `FocalStack`へ、`img`と`focusDistance`の要素が浮動小数点型であることを検証する処理を追加
- `FocalStack.makeAllImage`を、焦点スタック内の画像を先頭から順に結合してshape `(KHW, C)`の配列を返す処理へ変更
- `FocalStack.makeAllImage`の既知入出力テストと、初期化時の全プロパティおよび`allImg`を確認する統合テストを追加
- `FocalStack`の`None`、不適切な要素型、画素値範囲、光学値、画像枚数に対する異常系テスト9件を追加
- 既存の`depthFromDefocus`テストと合わせ、個別テスト3件、統合テスト2件、異常系テスト18件を1回で実行する構成へ変更

### `src/modules/myCameraCalibration.py`

- 動作確認用の`FocalStack`を、画像と光学パラメータを指定して初期化するよう変更
- `myExtrinsicParameterCalibration`の処理内容、引数、戻り値に変更なし

### 動作確認

- `.venv/bin/python`（Python 3.12.3、NumPy 1.26.4）で`depthFromDefocus.py`と`myCameraCalibration.py`の構文チェックに合格
- `FocalStack`と`src.modules.myCameraCalibration`のimportチェックに合格
- 個別テスト3件、統合テスト2件、異常系テスト18件の全23件に合格
- `src/modules/myCameraCalibration.py`の直接実行に合格し、期待する回転行列と並進ベクトルの出力を確認
- `FocalStack`の初期化と`makeAllImage`の処理フローについて、目視チェックリスト全5項目を確認

## v5.1.0（2026-10-20）

### `src/modules/depthFromDefocus.py`

#### 実装

- `FocalStack`の初期化処理を実装し、画像、焦点距離、合焦距離、レンズ開口径、ピクセルサイズをプロパティとして保持
- `FocalStack.makeAllImage`を実装し、焦点スタック内の画像を結合した`allImg`を生成
- 画素値が0以上255以下、各光学値が0より大きいこと、および画像枚数と合焦距離数が一致することを検証
- `depthFromDefocus`でも入力された`FocalStack`の数値条件を検証し、違反時に`ValueError`を送出
- `initAIFImage`の仮置き処理を、焦点スタックの平均から非空のAIF画像を返す処理へ変更
- `alternatingMinimization`の仮置き処理を、非空の深度マップとAIF画像を返す処理へ変更
- `initAIFImage`と`alternatingMinimization`の最適化アルゴリズムは未完成

#### テストコード

- テストを`個別テスト`、`統合テスト`、`異常系テスト`に分類し、分類ごとに採番
- 各テストの対象関数、概要、入力、実行コード、期待条件、期待出力、実際の出力、比較結果、合否を日本語で表示
- テスト用モックへ置き換えず、実際の関数へ既知入力を直接渡して期待出力と比較する方式へ変更
- 個別テストで`initAIFImage`と`alternatingMinimization`が空でない2次元配列を返すことを確認
- 統合テストで、shape `(1, 4)`の深度マップとshape `(4, 3)`のAIF画像について、型、shape、既知出力、不変条件を確認
- 異常系テストで、`None`、不適切な型1種類、画素値範囲、各光学値、画像枚数と合焦距離数の不一致を個別に確認

### 動作確認

- `.venv/bin/python`（Python 3.12.3、NumPy 1.26.4）で構文チェックと`src.modules.depthFromDefocus`のimportチェックに合格
- 個別テスト2件、統合テスト1件、異常系テスト9件の全12件に合格
- `depthFromDefocus`が`initAIFImage`、`alternatingMinimization`を順に呼び出し、深度マップとAIF画像を返す処理フローを目視確認
- テストコードの入力、期待条件、期待出力、実際の出力および合否判定をユーザーが目視確認

## v5.0.1（2026-10-20）

### `src/modules/myCameraCalibration.py`

- `myExtrinsicParameterCalibration`を実行し、問題なく動作することを確認
- 今後のコード変更時に出力を比較できるよう、正常動作時の回転行列と並進ベクトルを実行結果コメントとして追加
- 関数の処理内容、引数、戻り値に変更なし

### `src/other/calcurateCamPose.py`

- `calcurateCamPose.py`を実行し、問題なく動作することを確認
- 今後のコード変更時に出力を比較できるよう、正常動作時のカメラ間距離の計算値と回転行列の差分を実行結果コメントとして追加
- 計算処理と出力処理に変更なし

### 動作確認

- `src/modules/myCameraCalibration.py`を実行し、エラーなく期待した回転行列と並進ベクトルが出力されることを確認
- `src/other/calcurateCamPose.py`を実行し、エラーなくカメラ間距離の計算値と回転行列の差分が出力されることを確認

## v5.0.0（2026-10-19）

### `src/`

- Pythonコードを`src/`配下へ移動し、用途別のパッケージとして再編
- 再利用モジュールを`src/modules/`、評価コードを`src/evaluation/`、デモ・可視化・データ生成・補助処理を`src/other/`へ分類
- 各ディレクトリへ`__init__.py`を追加し、Pythonパッケージとしてimportできる構成へ変更

### `src/modules/`

- `depthFromDefocus.py`、`eightPointAlgorithm.py`、`eight_point_pose.py`、`myCameraCalibration.py`、`triangulate3DHPE.py`を再利用モジュールとして移動
- `myCameraCalibration.py`の内部importを、パッケージimportとファイル直接実行の両方に対応
- `triangulate3DHPE.py`へ型エイリアス、型ヒント、docstring、入力検証、明確な例外処理を追加
- Panoptic Datasetのパス生成、キャリブレーション読み込み、カメラパラメータ取得を独立した関数へ分離
- カメラパラメータとMediaPipe関連オブジェクトを必要になるまで読み込まない遅延初期化へ変更
- 8点アルゴリズムの処理を`eight_point_pose.py`へ委譲する互換関数を追加
- 入力画像、出力動画、推定ランドマーク、一時画像の既定パスを`data/`配下へ変更
- `depthFromDefocus.py`内の最適化処理と`eightPointAlgorithm.py`は仮置きのため未完成

### `src/evaluation/`

- 精度、MPJPE、部分精度、処理時間を評価する7ファイルを集約
- 評価コードが参照する入力・出力パスを`data/`配下へ変更

### `src/other/`

- カメラ較正、デモ、可視化、データ生成、補助処理、バックアップ用の12ファイルを集約
- `calcurateCamPose.py`と`triangulate3DHPE_show_result.py`から`src/modules/triangulate3DHPE.py`を参照できるようにimportを変更
- ハードコードされていた入出力パスを`data/`配下の相対パスへ変更
- `depthFromDefocus`の入力型と出力shapeを確認する動作確認用の`test.py`を追加

### 動作確認

- `src/`内の全Pythonファイルの構文チェックに合格
- `src.modules`内の主要5モジュールのimportに合格
- `src/modules/depthFromDefocus.py`の全10テストケースに合格
- `src/modules/myCameraCalibration.py`のファイル直接実行とパッケージ実行に合格
- 評価コード2ファイルで`is not 0`に関する`SyntaxWarning`を確認したが、構文チェックは合格

## v4.0.0（2026-10-19）

### `depthFromDefocus.py`

- `depthFromDefocus`関数を実装
- `focalStack`が`FocalStack`型でない場合に`TypeError`を送出する入力検証を追加
- `initAIFImage`と`alternatingMinimization`を順に呼び出し、2次元の深度マップと全焦点画像を返す処理を追加
- 最適化処理で使用するハイパーパラメータ定数を追加
- `FocalStack`クラスを仮置きとして追加（未完成）
- `initAIFImage`関数と`alternatingMinimization`関数を、空の2次元配列を返す仮置きとして追加（未完成）
- 仮置き関数の出力契約、関数間の呼び出し順、引数と戻り値の受け渡しを確認するテストを追加
- 深度が0以上、画像値が0以上255以下であることを確認する不変条件テストを追加
- `FocalStack`以外の7種類の入力で`TypeError`が発生することを確認するテストを追加
- 全10ケースの概要、入力、期待条件、実際の出力、合否を日本語で表示するテスト実行処理を追加

### `eightPointAlgorithm.py`

- `eightPointAlgorithm`関数の仮置きを追加
- 3次単位行列を回転行列、`[1, 0, 0]`を単位並進ベクトルとして返す処理を追加（未完成）

### `myCameraCalibration.py`

- `myExtrinsicParameterCalibration`関数を実装
- 対応点、焦点スタック、内部パラメータの型とshapeを検証する処理を追加
- 対応点が8点以上あり、2つの入力で点数が一致することを検証する処理を追加
- `eightPointAlgorithm`、`depthFromDefocus`、`recoverRealScaleTranslation`を呼び出し、回転行列と並進ベクトルを返す処理を追加
- `recoverRealScaleTranslation`関数を、単位並進ベクトルのコピーを返す仮置きとして追加（未完成）
- 8組の対応点、焦点スタック、内部パラメータを入力する動作確認用コードを追加

### 動作確認

- `depthFromDefocus.py`の構文チェックに合格
- `depthFromDefocus.py`の全10テストケースに合格
- `eightPointAlgorithm.py`と`myCameraCalibration.py`のテストは未実行

## v3.0.0

### `triangulate3DHPE_bestmp_accuracy.py`

- 各フレームで最も信頼度の高いランドマークとデータセットのランドマーク間のMPJPEを計算する処理を追加
- 信頼度のしきい値を指定できる機能を追加
- データセット内にしきい値未満のランドマークが存在する場合、そのフレームを評価対象から除外する処理を追加

### `triangulate3DHPE_bestmp_accuracy_conf90_perframe.py`

- 信頼度のしきい値以上のランドマークを対象に、最も信頼度の高いランドマークとデータセットのランドマーク間のMPJPEを計算する処理を追加
- フレーム全体ではなく、しきい値未満のランドマークのみを評価対象から除外する処理を追加

### `compute_mpjpe.py`

- 身体全体のMPJPEを計算する処理を追加

### `compute_mpjpe_copy.py`

- MPJPE計算処理の作成を開始（未完成）

## v2.0.0

### `triangulate3DHPE_partial_accuracy.py`

- 各フレームについて、両手および両足のMPJPEを表示する機能を追加

### `triangulate3DHPE_mpdata.py`

- 各フレームおよび各視点から、ランドマークごとに最も信頼度の高いデータを選択する処理を追加
- 選択したランドマークを`best_landmarks.json`へ保存する処理を追加

### `triangulate3DHPE_data.py`

- 推定ランドマークデータの保存形式を変更
- `[x0, y0, z0, x1, ..., z32]`形式から、`{"0": [x0, y0, z0], "1": [x1, y1, z1], ..., "32": [x32, y32, z32]}`形式の辞書へ変更

## v1.0.0

### `triangulate3DHPE_data.py`

- 推定したランドマークを集約する処理を追加

### `triangulate3DHPE_show_result.py`

- フレームごとの3次元プロットを表示する処理を追加

### `triangulate3DHPEcompute_mpjpe.py`

- 正解ランドマークと推定ランドマーク間のMPJPEを計算する処理を追加

### `triangulate3DHPE_partial_accuracy.py`

- 正解データに含まれないランドマークの精度を評価する処理を追加（未完成）

## v0.2.0

### 動画生成

- 2本の動画から連結画像の動画を生成する処理を追加

## v0.1.0

### 画像出力

- アノテーションあり画像とアノテーションなし画像を保存する処理を追加
- `concatenation_images`関数を追加
  - 出力画像を保存して連結する処理を追加

### ファイル構成

- 関連ファイルを`related_files`ディレクトリへ集約

## v0.0.0

### カメラ・データ入力

- HD動画を撮影する処理を追加
- データセットの内部パラメータと外部パラメータを入力する処理を追加

### 3次元プロット

- 3次元プロットの初期表示を調整
- 3次元プロットを保存する処理を追加
