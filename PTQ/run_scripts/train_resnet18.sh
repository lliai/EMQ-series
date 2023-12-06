#!/usr/bin/env bash

## resnet-18 3.0Mb
#CUDA_VISIBLE_DEVICES=2 python main_imagenet.py --data_path /home/stack/data_sdb/all_imagenet_data/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2]"
#
## resnet-18 3.5Mb
#python main_imagenet.py --data_path /home/stack/data_sdb/all_imagenet_data/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 3]"
#
## resnet-18 4.0Mb
#python main_imagenet.py --data_path /home/stack/data_sdb/all_imagenet_data/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 3, 3, 3]"

# resnet-18 4.5Mb
python main_imagenet.py --data_path /home/stack/data_sdb/all_imagenet_data/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3]"
#
## resnet-18 5.0Mb
#python main_imagenet.py --data_path /home/stack/data_sdb/all_imagenet_data/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 3, 3, 3, 4, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]"
#
## resnet-18 5.5Mb
#python main_imagenet.py --data_path /home/stack/data_sdb/all_imagenet_data/ --arch resnet18 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]"
