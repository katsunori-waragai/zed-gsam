#!/bin/bash
case $# in
  2)
    name=$1
    prompt=$2;;
  *)
    echo "usage:$0 name text_prompt"
    exit 1;;
esac

if [ ! -f ${name} ] ; then
   echo "not found ${name}"
   exit
fi
  
echo python3 gsam.py \
   --input_image ${name} \
   --output_dir "outputs" \
   --box_threshold 0.3   --text_threshold 0.25 \
   --text_prompt "${prompt}" \

