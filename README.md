# コーパスの前処理スクリプト

## 内容
`data/`: コーパス置き場
`kyoto_preprocess.py`: 京大コーパスの前処理スクリプト
`ntc_preprocess.py`: NAISTコーパスの前処理スクリプト
`bccwj_preprocess.py`: BCCWJコーパスの前処理スクリプト
`run.py`: 実行ファイル
`separator.py`: データを訓練用，開発用，評価用に分割するスクリプト

## 使い方
- `python run.py <corpus name>` で前処理を行う
- `python separator.py <corpus name>` でデータ分割を行う

## TODO
- それぞれのコーパスをどのようにモデルに渡すかを考える
