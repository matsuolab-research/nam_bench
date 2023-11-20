# nam_bench

## 目次
1. 動かし方
2. サポート範囲
      1. データセット
      2. メトリック
      3. コールバック
3. 拡張方法
     1. configの操作
     2. カスタムデータセット
     3. カスタムメトリック
     4. カスタムコールバック


# 1. 動かし方
基本的にはdemo.pyの通り.  
ユーザは、
* config_pathにconfigsディレクトリ以下のファイルの内から1つを指定  
(基本的にconfigファイルはユーザ自身で書き換えないでください)
* modelを定義(入力: torch.Tensor, 出力: torch.Tensor)
* output_dirを指定
以下のように評価をするクラスのインスタンス化&評価

```python
eval_op = eval_utils.Evaluation(model=model, config_path=config_path)
eval_op.evaluate(output_dir=output_dir)
```

評価結果はoutput_dirに"%Y%m%d%H%M%S"形式でディレクトリが作られ,result.csvとして各種表結果が保存されます.
callback関数がconfigに追加されている場合は,入出力結果もimgsディレクトリに保存されます.

# 2. サポート範囲
現状では,configファイルに指定可能なデータセット・メトリック・コールバック関数としては以下の通りになります.

* データセット
  * Digits: 0~9の分類
  * DigitsForInpainting: 1枚あたりに1つの数字が写っており,マスクされている範囲を予測.
  * MaskedMovingDigits: 1枚あたり1つの数字が写っており,n フレーム用意されている.0埋めされている最後のフレームの予測を行う.
* メトリック
  * 分類
    * accuracy, recall, precision, f1
  * 回帰
    * mse, mse_classwise, mse_metainfowise, mse_last_frame, mse_last_frame_classwise, mae
* コールバック関数
  * save_input_imgs, save_output_imgs, save_input_output_imgs

追加で欲しい、データセット・メトリック・コールバック関数があればお知らせください。


# 3. 拡張方法
## 3.1 configの操作
configのdataset_config以下のハイパーパラメータを変更することで,評価対象のデータセットを変更することができます.


## 3.2 カスタムデータセット
カスタムデータセットを追加する場合は,torch.utils.data.Datasetを継承したクラスを定義してください.  
継承したクラスのコンストラクタではconfigファイルのdataset_configのオブジェクトを受け取ります.  
生成したデータセットに対するメタ情報(マスクの位置,オブジェクトのスピード,etc...)はmetainfoという属性に,list[dict]として,各サンプルごとのメタ情報を格納してください.　　
必要に応じて,src/dataset/__init__.pyにカスタムデータセットの読み込みを追加してください.


## 3.3 カスタムメトリック
カスタムメトリックを追加する場合は,以下の要件を満たしてください.

* モデルへの入力および,モデルの出力,サンプル毎のメタ情報を受け取ります.
* ネストの無い辞書を返してください. e.g.) {"accuracy": 0.98}  

具体例については,src/metrics/common_metrics.pyを参照してください.


## 3.4 カスタムコールバック
モデルの入出力に対してコールバック関数を定義することができます.(再構成画像などを保存したい場合)  
具体的な例としては,src/callbacks/save_imgs.pyを参照してください.
