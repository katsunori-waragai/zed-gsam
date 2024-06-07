# changing and todo
- [x] --text_prompt で指定した内容に対して検出結果がないと、grounded_sam_demo.py は異常終了しないように対策する。
- [x] H, W をheight, widthの意味になるようにして、かつ従来通りのセグメンテーションの描画になるように修正する。
- [x] grounded_sam_output.jpg 中の原画像由来の領域を原画像の色順序に一致させる。
- [x] 保存するファイル名についての制約を減らす。保存する関数でのsignature の変更
- [x] changed output file names as follows
- [x] --image_dir を指定して入力フォルダ単位で処理するように改変した。
- [x] cap.py も複数の画像を保存できるよう改変した。
- [x] 後処理の時間がmatplotlibでかかりすぎているのを改善しよう。
- [x] torch.Tensor をわかりやすくする。
- [x] dataclass　を実装する。
- [x] `from some import *` はなくすこと。
- [x] 実行ディレクトリを制約しないように書き換えること
- [x] argsの処理をclass に反映させよう。
- [x] dino とsamの区別がつきやすい識別子にすること。
- [x] モジュールの外部で参照しないものは"_"始まりの変数名に変更する。
- [x] black を用いて書式をそろえたい。
  - project.toml にline length を記述した。
- [x] png ファイルも入力に受け付ける。 
- [x] use_sam_hq=Trueとすると、何が良くなるはずかを記載する。
  - 木製のイスをセグメンテーションしている事例がある。 
  - 標準のSAMの出力では、イスの隙間で地面の芝生が見えている領域までイスと同一のセグメンテーションになっている。 
  - SAM-HQ Outputでは、イスの隙間越しに見える芝生の領域の多くが、イスのセグメンテーションから外れている。
- [x] sam_hq_vit_h.pth をdownload して使えるようにすること
  - https://github.com/SysCV/sam-hq#model-checkpoints
  - gdown --fuzzy https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing
  - 自分用のOneDrive sam_hq_vit_h.pth
  - gdown --fuzzy https://drive.google.com/file/d/1XsTUFVy9o7vytZwf_zs-LR8UdZX1bmoU/view?usp=drive_link
  - huggingface からダウンロードできる。
  - https://huggingface.co/lkeab/hq-sam/tree/main
- [x] "SAM ViT version: vit_b / vit_l / vit_h" の違いは何か？
    - base, large, huge
      - https://zenn.dev/mattyamonaca/articles/dcacb4f6dcd58f
    - それらを変えた時のモデル*.pth はどこから入手するのか 
    - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
    - https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    - https://github.com/facebookresearch/segment-anything/issues/533
- https://github.com/SysCV/sam-hq?tab=readme-ov-file#model-checkpoints
- [x] SAM Vit version を軽量なモデルに切り替えることで高速化したい。
  - vit_b に切り替えた。
- [] zedhelper のインストール
- [] gsam_module のモジュールライブラリ化
- [] モジュールライブラリで使用すデータの所在の対応
- [] git clone を不要にしたい。
- [] pip でインストール可能なモジュールに自作モジュールを改変したい。
  - [] 連動して学習済みモデルのおく場所もpip でのインストール先に変更したい。
- [] testをきちんとtestにしよう。
- [] sam が標準のsamを使っているのをnanoSAMを使うように改変して処理時間を減らそう。
- [] ファイルへの保存なしという選択もできるようにAPIを変更しよう。
- [] grounding の処理時間は、２回め以降は1 [s] 以下になっている。
- [] PIL.Image はAPIのインタフェースから外す。
- [] --input_image を使用している従来のスクリプトが使えていない。
- [] モデルのtensorRT 化ができていない。
- [] 欠損値のある場合のdepth の表示を改善したい。
- [] zed.retrieve_image(depth_for_display, sl.VIEW.DEPTH); depth_for_display_cvimg = depth_for_display.get_data() で取得すると、欠損値が０になる。
  - 欠損値であることがわかりにくい。
  - zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH); depth_map_img = depth_map.get_data() のデータを確認すること。
  - こちらはshapeが（H,W)のデータ、値にNaNを含まない。(要確認)
- [] runtime_parameters.confidence_threshold を depth_and_gsam.py のコマンド引数で書き換えられるようにした。
- [] svoファイルを保存するツールをzed-sdk からコピーして追加。
  - ZED-SDK 4.1 からはsvo2のファイルが標準になっている。
  - そのため以下の手順で入力しても、svo2ファイルが保存された。
  - それを入力ファイルに指定して、自作スクリプトが動作した。
```commandline
python3 svo_recording.py -h
python3 svo_recording.py --output_svo_file bottles.svo
python3 svo_playback.py -h
python3 svo_playback.py --input_svo_file bottles.svo2 
python3 conf_and_depth.py --input_svo_file bottles.svo2
```

- [] svoファイルを入力として、自作ツールが動くようにする。

## conf_and_depth.py

疑問：
depth_map と pointsのcolor でisnanの比率が異なるのが原因不明である。
なお、depth_map と points_zとはisnanの比率が同等レベルである。

```commandline
runtime_parameters.confidence_threshold=100
runtime_parameters.enable_fill_mode=True
depth_map_data.shape=(1242, 2208) depth_map_data.dtype=dtype('float32') %
count_isfinite=2737038 99.807 %
count_isnan=0 0.000 %
count_isneginf=0 0.000 %
count_isposinf=5298 0.193 %

points.shape=(1242, 2208, 4)

count_isfinite_points=2013112  73.409 %
count_isnan_points=729224  26.591 %
count_isneginf_points=0  0.000 %
count_isposinf_points=0  0.000 %


count_isfinite_points_z=2737038  99.807 %
count_isnan_points_z=5298  0.193 %
count_isneginf_points_z=0  0.000 %
count_isposinf_points_z=0  0.000 %
```

