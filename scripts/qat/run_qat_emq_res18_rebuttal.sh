#!/bin/bash

# module load anaconda
# module load cuda/11.2
# module load cudnn/8.1.0.77_CUDA11.2
conda activate emq
echo "RUNNING with 8GPU with 4xbs 4xlr"
python QAT/quant_train.py \
 -a resnet18 \
 --epochs 90 \
 --lr 0.0016 \
 --batch_size 512 \
 --data /home/inspur/data/imagenet \
 --save_path ./output/qat_emq_resnet18_rebuttal/ \
 --act_range_momentum=0.99 \
 --wd 1e-4 \
 --data_percentage 1 \
 --pretrained \
 --fix_BN \
 --checkpoint_iter -1 \
 --distill_method KD_naive \
 --quant_scheme modelsize_6.69_a8_emq
