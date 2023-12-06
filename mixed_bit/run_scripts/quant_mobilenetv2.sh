#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "mobilenetv2" \
 --path "/Path/to/Basemodel" \
 --dataset "imagenet" \
 --save_path '/Path/to/Dataset' \
 --beta 0.000000001 \
 --model_size 0.9 \
 --quant_type "PTQ"
