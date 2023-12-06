#!/bin/bash
# Conduct experiments for searched bit width of activation and weight with EMQ for resnet18.
# NOTE: remember to check `n_bits_a`, `DATA_PATH`, `RESULT_PATH`

current_time=`date "+%Y_%m_%d"`

echo "Current time: $current_time"
DATA_PATH=/home/stack/data_sdb/all_imagenet_data/
RESULT_PATH=./output/official

mkdir -p $RESULT_PATH

# OMPQ

# resnet-18 4.5Mb
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3]"

# resnet-18 5.0Mb
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 3, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]"

# resnet-18 5.5Mb
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]"
