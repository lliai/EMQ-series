#!/usr/bin/env bash
python quant_train.py \
 -a resnet50 \
 --epochs 90 \
 --lr 0.0001 \
 --batch_size 128 \
 --data /Path/to/Dataset/ \
 --save_path /Path/to/Save_quant_model/ \
 --act_range_momentum=0.99 \
 --wd 1e-4 \
 --data_percentage 1 \
 --pretrained \
 --fix_BN \
 --checkpoint_iter -1 \
 --quant_scheme modelsize_16.0_a5_141BOP
