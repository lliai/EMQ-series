#!/usr/bin/env bash
python3 -m torch.distributed.launch --nproc_per_node=1 feature_extract.py \
 --model "resnet50" \
 --path "/Path/to/Basemodel" \
 --dataset "imagenet" \
 --save_path '/Path/to/Dataset' \
 --beta 3.3 \
 --model_size 16.0 \
 --quant_type "QAT"
