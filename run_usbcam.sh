#!/bin/bash
echo "start grounded Dino"
python3 gsam_movie.py \
   --output_dir "outputs_captured" \
   --box_threshold 0.3   --text_threshold 0.25 \
   --text_prompt "arm . cup . keyboard . table . plate . bottle . PC . person" \
