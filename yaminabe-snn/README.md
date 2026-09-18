# 闇鍋SNN — 最初の再帰LIFネットワーク

Python＋PyTorchによる独自実装です。入力 → 再帰LIF集団 → 非発火の読み出し器、という最小構成から始めます。初期設定のLIFは256個です。

接続構造、神経修飾、ニューロンモデルを別ファイルに分けました。今後はコネクトーム・神経修飾を組み込み、その後にニューロンモデルを改善していくための土台です。AlKilanyらの実装の複製や、論文の性能再現ではありません。

## 最初に動かす

ZIPを展開し、`yaminabe-snn`フォルダーをVS Codeで開いてください。以下のコマンドはすべて、このフォルダー内で実行します。uvが利用できる環境を想定しています。

まずCPUで動作確認する場合：

```bash
uv venv --python 3.11
uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install -e .
uv run --no-sync python -m yaminabe_snn.demo --device cpu
```

`--no-sync`は、上で用意した環境をそのまま利用する指定です。データのダウンロードは不要です。`runs/demo/dynamics.png`を開くと、次の内容を確認できます。

- 入力スパイク
- 256個のニューロンの発火時刻
- 代表ニューロンの膜電位と閾値
- 集団平均の発火率

同じ初期重み・同じ入力に対して、閾値を固定した条件と、80 ms以降に1.5倍にした条件を比較します。各試行の内部状態はゼロに戻します。この外部操作は調節インターフェースの動作確認であり、LC–NEモデルや性能改善の実証ではありません。

`runs/demo/traces.npz`に時系列、`summary.json`に設定と平均発火率を保存します。図の膜電位はリセット直前の値です。合成入力の同一ビンに複数イベントがある場合、入力図の点は重なりますが、計算にはイベント数を渡します。

## 次に小さな課題を学習する

```bash
uv run --no-sync python -m yaminabe_snn.train --dataset toy --epochs 10 --test --device cpu
```

合成スパイクの2クラス分類です。2つの入力群の活性化順序が異なります。最後の入力群だけでも解ける簡単な動作確認用課題で、時間記憶の能力やSHD性能を示すベンチマークではありません。

学習には代理勾配を使った時間方向の誤差逆伝播（BPTT）とAdamを使用します。e-propは未実装です。入力重み、再帰重み、読み出し重み・バイアスを学習し、閾値・時定数は初期版では固定します。学習中は調節器を無効にしています。

既定の目的関数は「交差エントロピー＋平均スパイク数／ステップ」です。`--spike-penalty`で後者の係数を変更できます（既定1.0、0で無効）。ログの`loss`は比較しやすいように交差エントロピーだけを記録します。発火率は別項目で記録します。平均スパイク数は消費電力そのものではありません。

学習用256サンプルを学習80％・検証20％に固定シードで分割し、検証の交差エントロピーが最小のチェックポイントを保存します。`--test`を付けた場合のみ、別シードで生成した128サンプルで最後に1回評価します。

| 出力 | 内容 |
|---|---|
| `runs/train/best.pt` | 重み、接続マスク、ネットワーク設定 |
| `runs/train/config.json` | 実行条件、使用環境、分割インデックス |
| `runs/train/history.json` | 各エポックの損失、精度、発火率 |
| `runs/train/learning.png` | 学習曲線 |
| `runs/train/summary.json` | 選択エポック、任意のテスト結果 |

同じ`--out`に再実行すると出力を上書きします。比較実験では`--out runs/experiment_name`を変えてください。

学習済みネットワークへの外部操作を表示する場合：

```bash
uv run --no-sync python -m yaminabe_snn.demo --checkpoint runs/train/best.pt --out runs/trained_demo
```

このデモはチェックポイントの種類にかかわらず合成入力を使います。SHDの評価は下記の学習コマンドで行います。

## SHDへ進む

[SHD配布元](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)からデータを取得して解凍し、以下の2ファイルを置いてください。

- `data/shd/shd_train.h5`
- `data/shd/shd_test.h5`

SHDは音声を人工的な聴覚モデルで700チャネルのスパイクに変換した、20クラスの分類データです。生体の実測スパイクではありません。

```bash
uv run --no-sync python -m yaminabe_snn.train --dataset shd --epochs 30 --batch-size 16 --test --out runs/shd
```

構成は700入力 → 256個の再帰LIF → 20出力です。`configs/shd.json`を自動で読みます。`--device auto`が既定で、CUDAが利用できればGPUを使います。GPU利用時は、お使いのGPU・ドライバーに対応するPyTorchを[公式インストール案内](https://pytorch.org/get-started/locally/)に従って環境に入れてください。CPU版のインストールだけではGPUは利用できません。

SHDの時刻は**秒**で読み、`dt_ms`に合わせてビンに変換します。同じビンの複数スパイクは個数を保持します。設定した記録時間外のイベントは切り捨て、個数を表示します。既定値は1 ms刻み・1,400 msです。時刻の単位・時間刻みを変えるときは入力側とモデル側をそろえてください。

公式の`shd_train.h5`内で学習・検証を分割します。`shd_test.h5`は最終評価にのみ使用します。現在の検証分割はサンプル単位のランダム分割で、話者ごとの分割ではありません。

この配布版では、SHDローダーを小さなHDF5検証データで確認しています。SHD本体を使った学習性能とGPUでの実行は未検証です。初期設定は性能を最適化したものではありません。

## ファイルの役割

| ファイル | 変更するもの |
|---|---|
| `yaminabe_snn/neurons.py` | LIFの更新式、発火とリセット、代理勾配 |
| `yaminabe_snn/model.py` | 入力・再帰集団・出力の組み合わせ、時間ループ |
| `yaminabe_snn/connectivity.py` | 接続の有無、再帰重み、接続行列の読み込み |
| `yaminabe_snn/modulation.py` | 閾値・膜時定数を変更するモジュール |
| `yaminabe_snn/data.py` | 合成データとSHDの入力変換 |
| `yaminabe_snn/train.py` | 学習、検証、チェックポイント保存 |
| `yaminabe_snn/demo.py` | 同じ入力での介入比較 |
| `configs/*.json` | ニューロン数、時定数、閾値、時間刻みなど |
| `tests/test_core.py` | 数値計算、接続方向、勾配、データ変換の確認 |

## 計算している式

膜電位は無次元で、静止電位0・初期閾値1とします。時間の単位はmsです。下付きtは離散的な時刻です。

```text
beta = exp(-dt / tau_syn)
alpha_t = exp(-dt / tau_mem_t)
gamma = exp(-dt / tau_readout)

W_mask = A * W_rec
W_effective[post, :] = W_mask[post, :] / max(1, sum(abs(W_mask[post, :])) / bound)
I_t = beta * I_(t-1) + W_in x_t + W_effective s_(t-1)
V_pre_t = alpha_t * V_(t-1) + (1 - alpha_t) * I_t
s_t = 1 if V_pre_t >= threshold_t else 0
V_t = V_pre_t - threshold_t * s_t
y_t = gamma * y_(t-1) + (1 - gamma) * (W_out s_t + bias)
```

- `A`は接続の有無を表す0/1マスク、`*`は要素ごとの積です。再帰入力には1ステップ前の発火を使います。
- 再帰による過剰な自己増幅を抑えるため、受け手ごとの再帰重みの絶対値合計に上限を設けます。`recurrent_row_bound`の既定値は2.0です。これは初期版の工学的な制約であり、生理学的な法則や全条件での安定性保証ではありません。JSONで`null`にすると無効になります。変更した場合は、精度と発火率の両方を確認してください。
- シナプスへのスパイク入力は電流の増分として加算し、その後指数減衰します。膜電位更新は、そのステップ内で電流を一定とする離散化です。
- 1ステップに発火は最大1回です。不応期、伝導遅延の分布、コンダクタンス型シナプスは未実装です。
- 代理勾配は`1 / (1 + 10 * abs(V_pre - threshold))**2`です。前向き計算は硬い閾値判定です。リセット項の勾配は切っています。
- 出力は最後の時刻の`y_t`です。確率ではなく、交差エントロピーに渡すスコアです。
- 閾値と膜時定数は`modulator`が返します。数値計算上は両者を最小値`1e-4`で制限します。独自調節器には別途、生理学的に妥当な範囲を設定してください。
- バッチごとに状態を初期化します。連続ストリーミング用の状態引き継ぎは未実装です。

## コネクトームへの入口

接続行列は**`A[受け手, 送り手]`**です。例えばニューロン0から1への接続は`A[1, 0] = 1`です。

小さな部分回路の0/1接続行列を`adjacency.npy`に保存すると、次のように読み込めます。

```bash
uv run --no-sync python -m yaminabe_snn.train --dataset toy --mask adjacency.npy --out runs/custom_graph
```

行列の形状は`n_hidden × n_hidden`、対角成分は0です。細胞数が異なる場合は、`--hidden N`またはJSONの`n_hidden`も合わせます。チェックポイントにはマスクも保存されます。禁止された接続は学習中も有効重みが0です。

初期版は接続行列を**密行列として計算**します。接続確率を下げてもメモリー・演算量は密行列のままです。ハエ全脳をそのまま読み込むための実装ではありません。次に部分回路のID対応・入力／出力細胞の指定を整備し、規模を増やす段階で疎行列やエッジ列による計算に切り替えます。

シナプス数をそのまま重みに換算する機能、伝達物質別の符号、Dale則、細胞座標・受容体分布は未実装です。初期重みには正負の制約がなく、初期値を除いて興奮性・抑制性ニューロンを区別していません。

## 神経修飾への入口

調節器は`nn.Module`で、以下の形を持ちます。

```python
def forward(self, time_ms, previous_spikes, threshold, tau_mem_ms):
    # previous_spikes: [batch, neuron]
    # 返り値は[neuron]または[batch, neuron]にブロードキャストできるテンソル
    return new_threshold, new_tau_mem_ms
```

最初は`NoModulation`、動作確認には`StepThreshold`を使います。生理学的な放出・濃度・受容体の状態を持つモジュールを追加するときは、**試行の境界でその状態もリセットする処理**を追加してください。現在の2種類は内部状態を持ちません。

## 確認コマンド

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

8項目を確認します。膜電位の解析解との一致、リセット、接続方向とマスクの勾配、再帰重みの上限、試行間の状態リセットと入力重みへの勾配、介入前の活動一致、SHDの秒単位変換・重複イベント・境界、小さなHDF5ファイルの読み込みです。

## 参照資料

- [AlKilanyらの公開実装](https://github.com/abdalalkilani/NeuromodulationSNN)：神経修飾SNNの設計上の参考。
- [SHDの仕様と配布元](https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/)：データの仕様。研究利用時は配布元の引用・ライセンス条件に従ってください。
- [PyTorchのカスタム自動微分](https://docs.pytorch.org/tutorials/beginner/examples_autograd/polynomial_custom_function.html)：代理勾配を定義する仕組み。

このコードの学習規則は工学的な最適化です。コネクトームや生理学的機構を追加しても、それだけで生体の学習則を再現したことにはなりません。

## 同梱した実行例

`examples/demo/`に可視化と時系列、`examples/toy_training/`に10エポックの合成データ学習結果とチェックポイントを収録しています。検証環境と確認範囲は`examples/verification.json`を参照してください。合成データの成績はSHDの成績ではありません。
