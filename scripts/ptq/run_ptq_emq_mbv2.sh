#!/bin/bash
# Conduct experiments for searched bit width of activation and weight with EMQ for resnet18.
# NOTE: remember to check `n_bits_a`, `DATA_PATH`, `RESULT_PATH`

current_time=`date "+%Y_%m_%d"`

echo "Current time: ${current_time}"
DATA_PATH=/home/stack/data_sdb/all_imagenet_data/
RESULT_PATH=./output/ptq_results
# SEED=888
SEED=222
# 888 denote the weight of 0.01; other are weight of 0.1
# 999 denote the weight of 1, is better than others
# 000 denote the weight of 1, rerun
# 111 denote the weight of 2,
# 222 denote the weight of 1, with _cfg for 1.3M
# 333 denote the weight of 1.5
# 444 denote the weight of 1, with _cfg for 1.5M
mkdir -p $RESULT_PATH


# EMQ
# mobilenetv2 1.3Mb activation: 8
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch mobilenetv2 --channel_wise --n_bits_a 8 --act_quant --weight 1 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 3, 4]" > $RESULT_PATH/run1_mobilenetv2_model_size1.3_${current_time}_${SEED}.log 2>&1 &



# mobilenetv2 1.5Mb activation: 8
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch mobilenetv2 --channel_wise --n_bits_a 8 --act_quant --weight 1.5 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4]" > $RESULT_PATH/run1_mobilenetv2_model_size1.5_${current_time}_${SEED}.log 2>&1 &


# candidate1
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch mobilenetv2 --channel_wise --n_bits_a 8 --act_quant --weight 1.5 --test_before_calibration --bit_cfg "[3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 3]" > $RESULT_PATH/run1_mobilenetv2_model_size1.5_${current_time}_candidate1.log 2>&1 &


# candidate2
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch mobilenetv2 --channel_wise --n_bits_a 8 --act_quant --weight 1.5 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]" > $RESULT_PATH/run1_mobilenetv2_model_size1.5_${current_time}_candidate2.log 2>&1 &

# candidate3
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch mobilenetv2 --channel_wise --n_bits_a 8 --act_quant --weight 1.5 --test_before_calibration --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3]" > $RESULT_PATH/run1_mobilenetv2_model_size1.5_${current_time}_candidate3.log 2>&1 &

# candidate4
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch mobilenetv2 --channel_wise --n_bits_a 8 --act_quant --weight 1.5 --test_before_calibration --bit_cfg "[3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 3]" > $RESULT_PATH/run1_mobilenetv2_model_size1.5_${current_time}_candidate4.log 2>&1 &
