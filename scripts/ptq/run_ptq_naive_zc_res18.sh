#!/bin/bash
# Conduct experiments for searched bit width of activation and weight with EMQ for resnet18.
# NOTE: remember to check `n_bits_a`, `DATA_PATH`, `RESULT_PATH`

current_time=`date "+%Y_%m_%d"`

echo "Current time: $current_time"
DATA_PATH=/home/stack/data_sdb/all_imagenet_data/
RESULT_PATH=./output/ptq_naive_zc
RUN=3
# RUN=2 only 4.5 and 4 model size are running for resnet18

mkdir -p $RESULT_PATH

# 5.5M
echo "handle hawqv1 5.5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_hwaqv1_${current_time}.log 2>&1 &

echo "handle hawqv2 5.5M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_hwaqv2_${current_time}.log 2>&1 &

echo "handle qe_score 5.5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_qe_score_${current_time}.log 2>&1 &

echo "handle bsnip 5.5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_bsnip_${current_time}.log 2>&1 &

echo "handle bsynflow 5.5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_bsynflow_${current_time}.log 2>&1

# 5M
echo "handle hawqv1 5M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_hwaqv1_${current_time}.log 2>&1 &

echo "handle hawqv2 5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_hwaqv2_${current_time}.log 2>&1 &

echo "handle qe_score 5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 3, 3, 4, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_qe_score_${current_time}.log 2>&1 &

echo "handle bsnip 5M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 3, 4, 4, 4, 3, 4, 3, 4, 4, 4, 3, 3, 4, 3, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_bsnip_${current_time}.log 2>&1 &

echo "handle bsynflow 5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_bsynflow_${current_time}.log 2>&1 &

# 4.5M
echo "handle hawqv1 4.5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_hwaqv1_${current_time}.log 2>&1

echo "handle hawqv2 4.5M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_hwaqv2_${current_time}.log 2>&1 &

echo "handle qe_score 4.5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 3, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_qe_score_${current_time}.log 2>&1 &

echo "handle bsnip 4.5M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_bsnip_${current_time}.log 2>&1 &

echo "handle bsynflow 4.5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 2, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_bsynflow_${current_time}.log 2>&1 &

# 4M
echo "handle hawqv1 4M"
CUDA_VISIBLE_DEVICES=3 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 2, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_hwaqv1_${current_time}.log 2>&1 &

echo "handle hawqv2 4M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 3, 4, 3, 3, 3, 3, 3, 4, 3, 2, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_hwaqv2_${current_time}.log 2>&1

echo "handle qe_score 4M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 3, 3, 4, 4, 2, 3, 2, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_qe_score_${current_time}.log 2>&1 &

echo "handle bsnip 4M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 2, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_bsnip_${current_time}.log 2>&1 &

echo "handle bsynflow 4M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 2, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_bsynflow_${current_time}.log 2>&1 &

# 3.5M
echo "handle hawqv1 3.5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_hwaqv1_${current_time}.log 2>&1 &

echo "handle hawqv2 3.5M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 3, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_hwaqv2_${current_time}.log 2>&1 &

echo "handle qe_score 3.5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_qe_score_${current_time}.log 2>&1

echo "handle bsnip 3.5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_bsnip_${current_time}.log 2>&1 &

echo "handle bsynflow 3.5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_bsynflow_${current_time}.log 2>&1 &

# 3M
echo "handle hawqv1 3M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_hwaqv1_${current_time}.log 2>&1 &

echo "handle hawqv2 3M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 2, 4, 4, 4, 4, 3, 4, 3, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_hwaqv2_${current_time}.log 2>&1 &

echo "handle qe_score 3M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_qe_score_${current_time}.log 2>&1 &

echo "handle bsnip 3M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_bsnip_${current_time}.log 2>&1 &

echo "handle bsynflow 3M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_bsynflow_${current_time}.log 2>&1 &
