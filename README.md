# nam_bench

# 目次
1. 動かし方
   1. インストール方法
   2. Evaluatorクラスの使い方
2. サポート範囲
   1. データセット
   2. メトリック
3. 拡張方法
   1. データセット
   2. メトリック



# 1. 動かし方
## 1.1 インストール方法
パッケージとして扱う場合は

```bash
git clone ...
cd ./nam_bench
git checkout sasaki-eval
pip install .
```

を実行することによって,demo_package.pyのように扱えます.  
一度パッケージとしてインストールすることができれば,細かいpathの設定をしなくて済むのでおすすめです.

## 1.2 Evaluatorクラスの使い方.
基本的には[demo_package.py](./demo_package.py)の通り.  
ユーザは、

* configsを元にEvaluatorのインスタンスのセットアップ
* Evaluatorに値を代入していくことによるセットアップ

の2通りが選べる.
config fileに基づくsetupに関しては[demo_package.py](./demo_package.py)の59行目付近を参照.

ここでは,Evaluatorに値を代入していくことによるセットアップについて簡単に説明する.  
全体の流れとしては以下の通り,
1. nam_bench.Evaluator()をインスタンス化する.
2. データセットを取得する関数の設定.
3. データセットの取得.
4. カスタムメトリックの設定(Optional)
5. 推論結果の評価


対応するコード
```python
eval_op = nam_bench.Evaluator()
    
# 登録されているデータセットであれば文字列を投げることで使用するデータセットを取得する関数が設定される.
# 登録されているデータセットには,デフォルトの評価項目が紐づけられている.
# 自作のデータセットを取得する関数を引数に渡すことも可能.
# ただし,返り値は
# {"train": {"X": data, "y": target, "metainfo": None|list[dict]}
# "test": {"X": data, "y": target, "metainfo": None|list[dict]}}とする必要あり.
eval_op.set_dataset_fn("MovingDigits")
x = eval_op.get_dataset(
    num_test_data=10*10,
    ... # 省略(呼び出す関数によって引数が変わる)
)

# カスタムメトリックの設定(任意)
# ただし,評価用の関数は引数に少なくとも(preds, labels, metainfo)を設定する必要あり.
eval_op.add_custom_eval_fn("mean", sample_custom_eval) # 名前, 評価関数

save_imgs_kwargs = {"save_dir": "./imgs", "preprocess_func": img_preprocess_fn}
eval_op.add_custom_eval_fn("save_imgs", nam_bench.metrics.make_imgs.save_imgs, **save_imgs_kwargs) # 名前, 評価関数, 評価関数の一部の引数

print(eval_op) # Evaluatorの設定状況が確認できる

# NOTE: ここの実装は各ユーザで行う. 入力・出力はnumpy.arrayにする必要あり
pred_y = inference(x)

# 評価の実行
reports = eval_op.evaluate(pred_y)

# reportsはOrderedDictになっており,{各評価項目: 評価結果}に加えて{metainfo: メタ情報}のキーが追加されている.
# この辞書をWandbなどのloggerに投げることを想定.
```


# 2. サポート範囲
現状では,configファイルに指定可能なデフォルトのデータセット・メトリックとしては以下の通りになります.

* データセット
  * MovingDigits: 1枚あたり1つの数字が写っており,n フレーム用意されている.0埋めされている最後のフレームの予測を行う.
* メトリック
  * 詳細は[common_metrics.py](./nam_bench/metrics/common_metrics.py)を参照

追加で欲しい、データセット・メトリックがあればお知らせください。


# 3. 拡張方法
## 3.1 データセット
一時的に使うのであれば,set_dataset_fnの引数にデータセットを読み込む関数を設定することをおすすめします.  
頻繁に使う場合は,[datasetsディレクトリ](./nam_bench/datasets)以下に新しいファイルを作り,
[__init__.py](./nam_bench/datasets/__init__.py)の```NAME2DATASETS_FN```の辞書に追加してください.  
ただし,辞書に登録する関数の返り値は以下の形式を要請します.(それ以外の値については実行時にエラーを吐きます)
```python
{
  "train": {"X": np.ndarray|None, "y": np.ndarray|None, "metainfo": None|list[dict]},
  "test": {"X": np.ndarray, "y": np.ndarray, "metainfo": None|list[dict]}
}
```

メタ情報はNoneでも構いませんが,メタ情報で集計して評価する際に便利なので,可能であれば設定しておくと良いと思います.  
生成したデータセットに対するメタ情報(マスクの位置,オブジェクトのスピード,etc...)はmetainfoという属性に,list[dict]として,各サンプルごとのメタ情報を格納してください.　　

また,登録したデータセットに紐づくデフォルトの評価項目を追加することができます.  
[const.py](./nam_bench/const.py)のDATASET2DEFAULT_EVALSに,データセットの名前とmetricの辞書が書かれています.  
設定可能なmetricは[__init__.py](./nam_bench/metrics/__init__.pyで呼び出せる関数群に限ります.


## 3.2 メトリック
一時的に使うメトリックに関しては,add_custom_eval_fnを通じて登録することをおすすめします.  
頻繁に使う場合はmetricは[metricsディレクトリ](./nam_bench/metrics)にファイルを追加することをおすすめします.  
メトリックは少なくとも(preds, labels, metainfo)を引数に持つ必要があります.  
それ以外の引数を設定することも可能ですが,add_custom_eval_fnの呼び出し時にキーワード引数として渡す必要があります.  
具体例に関しては,[common_metrics.py](./nam_bench/metrics/common_metrics.py)が参考になると思います.
