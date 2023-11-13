## 特徴量

- eRFMの特徴量（2017年以前のデータを用いて算出）
    - 最新購買日（Recency）
    - 総購買数（Frequency）
    - 利用期間（Tenure）
- 購買アイテム
    - 直前10アイテムを入れる

## 予測

- CLV（2018年のCV商品数）

## データセットの分割

### 学習データ

- 2017年のCLV、CV商品を予測
- 2016年以前でRFM、購買商品系列を計算

### 予測

- 2018年のCLV、CV商品を予測
- 2017年以前でRFM、購買商品系列を計算

## メモ

- RFM+LightGBMでどれくらい予測できるか見る

## 評価

- 推薦リストの作り方
    - randomに作って
    - スコアの総和をとる

## 今後の方針

- CLV予測の精度を上げる
- 位置エンコーディングは必要？
    - 最後のものを取り出しているなら必要なさそう
- 離反の0-1予測の方が良いかも
- ランダムサンプリングだと弱いのかも
- 直近の購買履歴を見れないのはやっぱり弱いか？
- context_itemsに関して
    - ランダムに選ぶのではなく、時系列で普通に選ぶ？
    - ランダムに選んだ後に時系列順に並び直す？
- 推薦結果に対して、多様性の指標を計算する

## リファクタ

- CLV -> 適切な名前
- churn -> alive
- 離反 -> 生存
- Datasetのfeatureをuser_idx、sequenceで分ける

## 評価

- アイテムを入れないで離反予測
- アイテムだけで離反予測

## 結果の記録

- "out/rec.csv"
    ```
    before_rerank_size = 50
    filter_sample_size = 10
    rerank_sample_size = 50
    top_k = 10
    ```

- "out/rec2.csv"
    ```
    before_rerank_size = 20
    filter_sample_size = 10
    rerank_sample_size = 10
    top_k = 10
    ```

- "out/rec3.csv"
    ```
    leaky_relu
    before_rerank_size = 50
    filter_sample_size = 50
    rerank_sample_size = 50
    top_k = 10
    ```

- "out/rec4.csv"
    ```
    leaky_relu+pe
    before_rerank_size = 50
    filter_sample_size = 50
    rerank_sample_size = 50
    top_k = 10
    ```

- "out/rec5.csv"
    ```
    leaky_relu+pe@large
    before_rerank_size = 50
    filter_sample_size = 50
    rerank_sample_size = 50
    top_k = 10
    ```