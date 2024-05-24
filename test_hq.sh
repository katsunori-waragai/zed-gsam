#!/bin/bash
IMAGE_HQ_DIR=image_hq
mkdir -p ${IMAGE_HQ_DIR}
cd ${IMAGE_HQ_DIR} && wget https://github.com/IDEA-Research/detrex-storage/releases/download/grounded-sam-storage/sam_hq_demo_image.png
cd ..

export CUDA_VISIBLE_DEVICES=0
python3 gsam.py \
   --use_sam_hq \
   --image_dir ${IMAGE_HQ_DIR} \
   --output_dir "outputs_${IMAGE_HQ_DIR}" \
   --box_threshold 0.3   --text_threshold 0.25 \
   --text_prompt "arm . cup . keyboard . table . chair" \
