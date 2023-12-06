#!/bin/bash

# for QAT
# CUDA_VISIBLE_DEVICES=0 python tests/test_constraint_size.py --arch 'resnet18' --model_size 6.7 --quant_type PTQ 2>&1 &

CUDA_VISIBLE_DEVICES=0 python tests/test_constraint_size.py --arch 'resnet50' --model_size 16 --quant_type PTQ 2>&1 &

CUDA_VISIBLE_DEVICES=1 python tests/test_constraint_size.py --arch 'resnet50' --model_size 18 --quant_type PTQ 2>&1 &

CUDA_VISIBLE_DEVICES=0 python tests/test_constraint_size.py --arch 'resnet50' --model_size 21 --quant_type PTQ 2>&1 &
