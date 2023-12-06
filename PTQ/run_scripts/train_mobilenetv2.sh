#!/usr/bin/env bash

## mobilenetv2 1.5Mb
#python main_imagenet.py --data_path /Path/to/Dataset/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]"

# mobilenetv2 1.3Mb
python main_imagenet.py --data_path /Path/to/Dataset/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 2]"

## mobilenetv2 1.1Mb
#python main_imagenet.py --data_path /Path/to/Dataset/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 2, 2, 2]"
#
## mobilenetv2 0.9Mb
#python main_imagenet.py --data_path /Path/to/Dataset/ --arch mobilenetv2 --n_bits_w 2 --channel_wise --n_bits_a 8 --act_quant --weight 0.1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 2]"
