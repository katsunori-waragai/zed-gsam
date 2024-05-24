## grounded-SAM docker
- docker environment for grounded SAM
https://github.com/IDEA-Research/Grounded-Segment-Anything

### Dockerファイル内での処理
- GroundedSAMを使うための環境構築
- GroundedSAMを使うためのpre-trained file のダウンロード
- ユーザー作成ファイルのCOPY

### status
sh docker_build.sh 
sh docker_run.sh
succeeded.

## usage
### test_cap_and_demo.sh
USBカメラから画像を取得・保存して、その画像に対して、grounded-SAMのdemo相当の処理を行う。

### run_usbcam.sh
- USBカメラ　入力でのセグメンテーション


### test_pre-captured.sh
- capture済の画像をセグメンテーションする。

### gsam.py
```commandline
python3 gsam.py -h
usage: gsam.py [-h] [--use_sam_hq] --image_dir IMAGE_DIR --text_prompt TEXT_PROMPT --output_dir OUTPUT_DIR
                                      [--box_threshold BOX_THRESHOLD] [--text_threshold TEXT_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --use_sam_hq          using sam-hq for prediction
  --image_dir IMAGE_DIR
                        path to image file
  --text_prompt TEXT_PROMPT
                        text prompt
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        output directory
  --box_threshold BOX_THRESHOLD
                        box threshold
  --text_threshold TEXT_THRESHOLD
                        text threshold

python3 gsam_movie.py -h
usage: gsam_movie.py [-h] [--use_sam_hq] --text_prompt TEXT_PROMPT --output_dir OUTPUT_DIR [--box_threshold BOX_THRESHOLD]
                     [--text_threshold TEXT_THRESHOLD]

Grounded-Segment-Anything for USB camera

optional arguments:
  -h, --help            show this help message and exit
  --use_sam_hq          using sam-hq for prediction
  --text_prompt TEXT_PROMPT
                        text prompt
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        output directory
  --box_threshold BOX_THRESHOLD
                        box threshold
  --text_threshold TEXT_THRESHOLD
                        text threshold

```

## output files
```commandline
outputs/demo1_mask.jpg
outputs/demo1_mask.json
outputs/demo1_raw.jpg
outputs/demo1_sam.jpg
```

## SAM model selection
### execution time
```commandline
vit_h case
used_time={'dino': 4.601885602, 'sam': 2.78923419, 'save_mask': 0.223663634, 'save_sam': 0.101654286}
used_time={'dino': 0.619241372, 'sam': 1.836635902, 'save_mask': 0.221149576, 'save_sam': 0.099712454}
used_time={'dino': 0.509424686, 'sam': 1.837601291, 'save_mask': 0.193354196, 'save_sam': 0.098929636}
used_time={'dino': 0.513645634, 'sam': 1.833863588, 'save_mask': 0.221209674, 'save_sam': 0.099474151}
used_time={'dino': 0.583415437, 'sam': 1.867050265, 'save_mask': 0.205879691, 'save_sam': 0.099982281}
used_time={'dino': 0.516912789, 'sam': 1.831642283, 'save_mask': 0.183015659, 'save_sam': 0.103907674}
vit_l case
used_time={'dino': 4.801369182, 'sam': 1.433056724, 'save_mask': 0.238014577, 'save_sam': 0.090027557}
used_time={'dino': 0.581472926, 'sam': 1.070962919, 'save_mask': 0.237833933, 'save_sam': 0.088467575}
used_time={'dino': 0.513170818, 'sam': 1.073418961, 'save_mask': 0.21821957, 'save_sam': 0.087076492}
used_time={'dino': 0.517417823, 'sam': 1.097062882, 'save_mask': 0.237811434, 'save_sam': 0.227442488}
used_time={'dino': 0.512869719, 'sam': 1.089623328, 'save_mask': 0.218557809, 'save_sam': 0.087464781}
used_time={'dino': 0.518495263, 'sam': 1.090717183, 'save_mask': 0.196923878, 'save_sam': 0.086473765}
```

### segmentation quality
- 

## todo
- use stable opencv-python


