#!/bin/bash
# Conduct series experiments for different search space structure
# Avaliable structure: linear, tree, and graph
SEARCH_STRUCTURES="tree graph linear"
# graph
# linear

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH=/home/stack/data_sdb/all_imagenet_data/
SAVE_DIR=./output/rnd_search_emq_zc
ITERATIONS=1000
# SEEDs="123 234 345 456"
SEEDs="123 234 345 456 987 876 765 654 543"
# 888 999"

T="$(date +%s)"

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# unloop above scripts
# SEARCH_STRUCTURE='tree'
# CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_${CURRENT_TIME}_${SEED}.log  2>&1 &

# SEARCH_STRUCTURE='graph'
# CUDA_VISIBLE_DEVICES=2 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_${CURRENT_TIME}_${SEED}.log 2>&1 &

# SEARCH_STRUCTURE='linear'
# CUDA_VISIBLE_DEVICES=1 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_${CURRENT_TIME}_${SEED}.log 2>&1 &

for SEED in ${SEEDs}
    do
        BS=8
        CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18 --batch_size ${BS}  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_BS_${BS}_${CURRENT_TIME}_${SEED}.log  2>&1

        BS=16
        CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18 --batch_size ${BS}  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_BS_${BS}_${CURRENT_TIME}_${SEED}.log  2>&1


        BS=32
        CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18 --batch_size ${BS}  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_BS_${BS}_${CURRENT_TIME}_${SEED}.log  2>&1


        SEARCH_STRUCTURE='tree'
        BS=64
        CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18 --batch_size ${BS}  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_BS_${BS}_${CURRENT_TIME}_${SEED}.log  2>&1


        SEARCH_STRUCTURE='tree'
        BS=128
        CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18 --batch_size ${BS}  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_BS_${BS}_${CURRENT_TIME}_${SEED}.log  2>&1

        SEARCH_STRUCTURE='tree'
        BS=256
        CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_emq_zc.py --arch resnet18 --batch_size ${BS}  --data_path ${DATA_PATH}  --weight 0.01  --seed ${SEED} > ${SAVE_DIR}/rnd_search_emq_zc_${SEARCH_STRUCTURE}_resnet18_BS_${BS}_${CURRENT_TIME}_${SEED}.log  2>&1

    done
# echo total time
T="$(($(date +%s)-T))"
echo "It took ${T} seconds!"
