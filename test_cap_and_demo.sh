#!/bin/bash
python3 cap.py --is_zed
python3 gsam.py \
   --image_dir captured \
   --output_dir "outputs_captured" \
   --box_threshold 0.3   --text_threshold 0.25 \
   --text_prompt "arm . cup . keyboard . table " \
