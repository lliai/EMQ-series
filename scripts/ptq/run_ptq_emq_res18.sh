#!/bin/bash
# Conduct experiments for searched bit width of activation and weight with EMQ for resnet18.
# NOTE: remember to check `n_bits_a`, `DATA_PATH`, `RESULT_PATH`

current_time=`date "+%Y_%m_%d"`

echo "Current time: $current_time"
DATA_PATH=/home/stack/data_sdb/all_imagenet_data/
RESULT_PATH=./output/run_ptq_emq_res18
RUN=5
# RUN=2 only 4.5 and 4 model size are running for resnet18

mkdir -p $RESULT_PATH


# EMQ
# # resnet18 4.5Mb activation: 8
# echo "resnet18 4.5M A8"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 3, 3, 3, 3, 4]" > $RESULT_PATH/run${RUN}_resnet18_model_size4.5_${current_time}.log 2>&1 &

# # # # resnet18 5.5Mb activation: 4
# echo "resnet18 5.5M A4"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 4 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]" > $RESULT_PATH/run${RUN}_resnet18_model_size5.5_${current_time}.log 2>&1 &

# # resnet18 4Mb activation: 8
# echo "resnet18 4M A8"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 2, 2, 3, 3, 4]" > $RESULT_PATH/run${RUN}_resnet18_model_size4_${current_time}.log 2>&1 &



# # resnet18 test full-precision
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]" > $RESULT_PATH/run${RUN}_resnet18_model_size_${current_time}.log


# RUN=666

# # 5.5M
# echo "handle emq 5.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_emq_${current_time}.log 2>&1

# # 5M
# echo "handle emq 5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_emq_${current_time}.log 2>&1 &

# # 4.5M
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1

# # 4M
# echo "handle emq 4M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 2, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_emq_${current_time}.log 2>&1 &

# # 3.5M
# echo "handle emq 3.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_emq_${current_time}.log 2>&1


# # 3M
# echo "handle emq 3M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_emq_${current_time}.log 2>&1 &


# RUN=777

# # 5.5M
# echo "handle emq 5.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_emq_${current_time}.log 2>&1

# # 5M
# echo "handle emq 5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_emq_${current_time}.log 2>&1 &

# # 4.5M
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1

# # 4M
# echo "handle emq 4M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 2, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_emq_${current_time}.log 2>&1 &

# # 3.5M
# echo "handle emq 3.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_emq_${current_time}.log 2>&1


# # 3M
# echo "handle emq 3M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_emq_${current_time}.log 2>&1 &

RUN=111
# 222 denotes the weight of 1

# # 5.5M
# echo "handle emq 5.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5.5_emq_${current_time}.log 2>&1 &

# # 5M
# echo "handle emq 5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 3, 4, 3, 4, 3, 4, 3, 3, 4, 4, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size5_emq_${current_time}.log 2>&1 &

# # 4.5M
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &

# # 4M
# echo "handle emq 4M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 4, 3, 2, 2, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4_emq_${current_time}.log 2>&1 &

# # 3.5M
# echo "handle emq 3.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 3, 4, 2, 4, 4, 4, 4, 4, 3, 3, 4, 3, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3.5_emq_${current_time}.log 2>&1 &


# # 3M
# echo "handle emq 3M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2]" >> $RESULT_PATH/run${RUN}_resnet18_model_size3_emq_${current_time}.log 2>&1 &


# RUN=1111
# # 222 denotes the weight of 1
# # 4.5M 69.66
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &

# RUN=2222
# # 222 denotes the weight of 1.5
# # 4.5M 69.18
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1.5 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=3333
# # 222 denotes the weight of 2
# # 4.5M 69.34
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 2 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &

# RUN=4444
# # 222 denotes the weight of 0.5
# # 4.5M 69.37
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 0.5 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &

# RUN=5555
# # 222 denotes the weight of 0.1
# # 4.5M 69.49
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 0.1 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=6666
# # 6666 denotes the weight of 1 run 1
# # 4.5M 69.46
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=7777
# # 7777 denotes the weight of 1 run 2
# # 4.5M 69.45
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=8888
# # 8888 denotes the weight of 1 run 3
# # 4.5M 69.66
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 2, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# another bit_cfg

# RUN=00000
# # 6666 denotes the weight of 1 run 1
# # 4.5M 69.49
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=11111
# # 7777 denotes the weight of 1 run 2
# # 4.5M 69.35
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=22222
# # 8888 denotes the weight of 1 run 3
# # 4.5M  69.45
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=44444
# # 6666 denotes the weight of 1 run 1
# # 4.5M 69.44
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=55555
# # 7777 denotes the weight of 1 run 2
# # 4.5M 69.44
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


# RUN=66666
# # 8888 denotes the weight of 1 run 3
# # 4.5M 69.36
# echo "handle emq 4.5M"
# CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 1 --bit_cfg "[4, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 4, 4, 3, 3, 3, 3]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


RUN=668
# 8888 denotes the weight of 1 run 3
# 4.5M
echo "handle emq 4.5M"
CUDA_VISIBLE_DEVICES=0 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 3 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &

RUN=686
# 8888 denotes the weight of 1 run 3
# 4.5M
echo "handle emq 4.5M"
CUDA_VISIBLE_DEVICES=1 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 3 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &


RUN=866
# 8888 denotes the weight of 1 run 3
# 4.5M
echo "handle emq 4.5M"
CUDA_VISIBLE_DEVICES=2 python PTQ/main_imagenet.py --data_path $DATA_PATH --arch resnet18 --channel_wise --n_bits_a 8 --act_quant --test_before_calibration --n_bits_w 2 --weight 3 --bit_cfg "[4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 3, 3, 4, 4, 3, 3, 3, 4]" >> $RESULT_PATH/run${RUN}_resnet18_model_size4.5_emq_${current_time}.log 2>&1 &
