# 独自クラス仕様

本ファイルは、プロジェクト内で共通して使用する独自クラスの仕様を定義する。
各関数の仕様書ではクラスの定義を重複して記載せず、本ファイルの該当クラスを参照する。

## 共通規則

- 配列型には`np.ndarray`を使用する。
- shapeの各アルファベットは以下の内容を示す。
    - `K`は焦点スタックの画像枚数
    - `H`は画像の高さ
    - `W`は画像の幅
    - `C`は画像のチャネル数（RGB画像なら3）
- 入力される画像は全て同じサイズである。
- 長さの単位はメートルとする。
- 未決定の項目は推測せず空欄にする。
- 表の`-`は関連しない項目であることを示す。
- クラス名はパスカルケース、変数名や関数名はキャメルケースで記述する

## クラス一覧

|クラス名|配置ファイル|概要|
|---|---|---|
|`FocalStack`|`depthFromDefocus.py`|焦点スタックの各画像と撮影時の光学パラメータを保持する|
|`ExtrinsicParameterMats`|`eightPointAlgorithm.py`|回転行列と並進ベクトルを保持する|

# `FocalStack` クラス

## 概要

焦点スタックの各画像と撮影時の光学パラメータを保持する。
対象となる光学パラメータは焦点距離、各焦点スタック画像の合焦距離、レンズ開口径、ピクセルの実寸幅の4つである。

## 配置・インポート
- 配置ファイル: `depthFromDefocus.py`
- インポート方法: 
```python
from depthFromDefocus import FocalStack
```

## プロパティ
|プロパティ名|型|shape|要素の型|単位|説明|
|---|---|---|---|---|---|
|img|nd.array|(K, HW, C)|float|-|H×WのCチャネルを持つ画像を画素ごとに変換した行列|
|allImg|nd.array|(KHW, C)|float|-|img内の各画像の画素値を一列に変換した行列
|focalLength|float|-|-|m|焦点距離|
|focusDistance|np.ndarray|(K,)|float|m|各画像の物体側合焦距離|
|apertureDiameter|float|-|-|m|レンズ開口径|
|pixelSize|float|-|-|m/pixel|1ピクセルの実寸幅|

## 初期化処理
1. 引数の型がプロパティと正しいかを検証する（ただし、shapeに含まれるアルファベットは考慮しない）
    - `img`と`focusDistance`の要素の型は浮動小数点型とする
2. `img`の全画素値が0以上255以下であることを検証する
3. `focalLength`、`focusDistance`の全要素、`apertureDiameter`、`pixelSize`が0より大きいことを検証する
4. `img`の画像枚数と`focusDistance`の要素数が一致することを検証する
5. 数値条件に違反した場合は`ValueError`を送出する
6. `allImg`は`makeAllImage`メソッドの戻り値を代入
7. 対応するインスタンスプロパティに保存

## makeAllImage メソッド 
`self.img`内のK枚の(HW, C)の画像を一列に直す。
最終的に(KHW, C)の行列を返す。

入力
|引数名|型|shape|要素の型|説明|
|---|---|---|---|---|
|self|FocalStack|-|-|`img`プロパティを持つ自身のインスタンス|

出力
|戻り値名|型|shape|要素の型|説明|
|---|---|---|---|---|
|allImage|np.ndarray|(KHW, C)|float|`self.img`内のK枚の画像を一列に変換した行列|

処理フロー
1. 空の配列`J`を作成する
2. 以下をK回ループ
3. `self.img[i]`を取得する
4. 列方向に`self.img[i]`を`J`に追加する

## テスト方法

### 個別テスト

- `makeAllImage`へ2枚の既知画像を持つ`FocalStack`を入力する。
- 画像が入力順に結合され、shape `(8, 3)`の既知出力と一致することを確認する。

### 統合テスト

- 既知の`img`、`focalLength`、`focusDistance`、`apertureDiameter`、`pixelSize`で`FocalStack`を初期化する。
- 全入力が対応するプロパティへ保存され、`allImg`が既知出力と一致することを確認する。

### 異常系テスト

- `img`へ`None`を入力した場合に`TypeError`が発生することを確認する。
- `img`の要素が整数型の場合に`TypeError`が発生することを確認する。
- 画素値が0未満または255を超える場合に`ValueError`が発生することを確認する。
- 焦点距離、合焦距離、レンズ開口径、ピクセルサイズが0以下の場合に、それぞれ`ValueError`が発生することを確認する。
- 画像枚数と合焦距離数が一致しない場合に`ValueError`が発生することを確認する。

### 処理フロー目視チェックリスト

型、shape、値域、要素数および具体的な出力値は自動テストで検証し、このチェックリストには含めない。

- [ ] 初期化時に`img`を保存してから`makeAllImage`を呼び出している。
- [ ] `makeAllImage`の戻り値を`allImg`へ保存している。
- [ ] `makeAllImage`が`self.img`を先頭から順番に処理している。
- [ ] 各画像を処理するループに、画像の取得と結合処理が欠けなく含まれている。
- [ ] 最後に結合済みの画像を返している。

