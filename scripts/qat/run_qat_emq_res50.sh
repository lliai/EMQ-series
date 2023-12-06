#!/bin/bash

# module load anaconda
# module load cuda/11.2
# module load cudnn/8.1.0.77_CUDA11.2
# source activate py38

python QAT/quant_train.py \
 -a resnet50 \
 --epochs 90 \
 --lr 0.0001 \
 --batch_size 128 \
 --data /home/stack/data_sdb/all_imagenet_data/ \
 --save_path ./output/qat_emq_resnet50/ \
 --act_range_momentum=0.99 \
 --wd 1e-4 \
 --data_percentage 1 \
 --pretrained \
 --fix_BN \
 --checkpoint_iter -1 \
 --quant_scheme bit_config_resnet50_emq_17_86
