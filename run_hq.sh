#!/bin/bash
wget https://github.com/IDEA-Research/detrex-storage/releases/download/grounded-sam-storage/sam_hq_demo_image.png

export CUDA_VISIBLE_DEVICES=0
python3 grounded_sam_demo.py \
  --config GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --grounded_checkpoint groundingdino_swint_ogc.pth \
  --sam_hq_checkpoint ./sam_hq_vit_h.pth \
  --use_sam_hq \
  --input_image sam_hq_demo_image.png \
  --output_dir "outputs_hq" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --text_prompt "chair." \
  --device "cuda"
