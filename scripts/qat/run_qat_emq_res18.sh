#!/bin/bash

# module load anaconda
# module load cuda/11.2
# module load cudnn/8.1.0.77_CUDA11.2
source activate py38

CUDA_VISIBLE_DEVICES=0,1,2 python quant_train.py \
 -a resnet18 \
 --epochs 90 \
 --lr 0.0002 \
 --batch_size 128 \
 --data /home/stack/data_sdb/all_imagenet_data \
 --save_path ./checkpoints/imagenet/test/ \
 --act_range_momentum=0.99 \
 --wd 1e-4 \
 --data_percentage 1 \
 --fix_BN \
 --checkpoint_iter -1 \
 --quant_scheme modelsize_6.7_a6_75B
