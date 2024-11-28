# ZED SDK との連動がないスクリプトの実行例

以下のpythonスクリプトは、USBカメラ入力に対して、--text_prompt　に示される対象物に対して
インスタンスセグメンテーションを実施して、--output_dir に保存するスクリプトである。


```commandline
Grounded-Segment-Anything# python3 gsam_movie.py --text_prompt bottle --output_dir outputs
```

![captured_0001_raw.jpg](figures/gsam_figures/captured_0001_raw.jpg
![captured_0001_mask.jpg](figures/gsam_figures/captured_0001_mask.jpg)

![captured_0001_sam.jpg](figures/gsam_figures/captured_0001_sam.jpg)

```:captured_0001_mask.json

[{"value": 0, "label": "background"}, {"value": 1, "label": "bottle", "logit": 0.72, "box": [776.5209350585938, 523.3881225585938, 861.9423217773438, 747.7156982421875]}, {"value": 2, "label": "bottle", "logit": 0.7, "box": [508.3803405761719, 653.546630859375, 621.994873046875, 880.4595336914062]}, {"value": 3, "label": "bottle", "logit": 0.75, "box": [882.2781372070312, 497.0130615234375, 957.2284545898438, 703.328369140625]}, {"value": 4, "label": "bottle", "logit": 0.74, "box": [649.7821044921875, 560.226318359375, 755.00244140625, 804.9031982421875]}]
```

-----
![captured_0002_raw.jpg](figures/gsam_figures/captured_0002_raw.jpg)
![captured_0002_mask.jpg](figures/gsam_figures/captured_0002_mask.jpg)
![captured_0002_sam.jpg](figures/gsam_figures/captured_0002_sam.jpg)

```:captured_0002_mask.json
[{"value": 0, "label": "background"}, {"value": 1, "label": "bottle", "logit": 0.66, "box": [882.1499633789062, 496.5115051269531, 957.7581787109375, 703.1181640625]}, {"value": 2, "label": "bottle", "logit": 0.68, "box": [775.8727416992188, 522.694091796875, 855.7272338867188, 747.4675903320312]}, {"value": 3, "label": "bottle", "logit": 0.69, "box": [619.7833251953125, 559.8886108398438, 727.797119140625, 805.8375244140625]}, {"value": 4, "label": "bottle", "logit": 0.68, "box": [507.5815734863281, 652.99072265625, 622.3192749023438, 880.9000244140625]}, {"value": 5, "label": "cup", "logit": 0.69, "box": [825.576904296875, 678.4196166992188, 922.5826416015625, 811.3165893554688]}, {"value": 6, "label": "book", "logit": 0.35, "box": [973.5555419921875, 671.3574829101562, 1206.4371337890625, 756.1993408203125]}, {"value": 7, "label": "book", "logit": 0.37, "box": [972.8187255859375, 634.828125, 1207.2337646484375, 757.08203125]}]
```