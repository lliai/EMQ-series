#!/bin/bash

# available zc for bit: 'hawqv1', 'hawqv2', 'orm',
#       'bparams', 'bn_score', 'bsnip', 'bsynflow'

ZC_CANDIDATES="orm hawqv1 hawqv2 qe_score bparams bsynflow bsnip"
#

CURRENT_TIME=`date "+%Y_%m_%d"`
DATA_PATH=/home/stack/data_sdb/all_imagenet_data/
SAVE_DIR=./output/rank_naive_zc
SAMPLES=50
SEED=6686
ARCH=resnet18
# 888 999"

T="$(date +%s)"

mkdir -p ${SAVE_DIR}
echo "Current time: $CURRENT_TIME"

# orm
ZC=orm
echo "Current zc: ${ZC}"
CUDA_VISIBLE_DEVICES=1 python exps/rank/rank_mqbench/rank_zc_w_bit.py --data_path ${DATA_PATH} --zc_name ${ZC} --samples ${SAMPLES} --seed ${SEED} --arch ${ARCH} > ${SAVE_DIR}/rank_naive_zc_kd_sp_ps_${ZC}_${SAMPLES}_${SEED}.log 2>&1 &

# hawqv1
ZC=hawqv1
echo "Current zc: ${ZC}"
CUDA_VISIBLE_DEVICES=1 python exps/rank/rank_mqbench/rank_zc_w_bit.py --data_path ${DATA_PATH} --zc_name ${ZC} --samples ${SAMPLES} --seed ${SEED} --arch ${ARCH} > ${SAVE_DIR}/rank_naive_zc_kd_sp_ps_${ZC}_${SAMPLES}_${SEED}.log 2>&1 &

# hawqv2
ZC=hawqv2
echo "Current zc: ${ZC}"
CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_zc_w_bit.py --data_path ${DATA_PATH} --zc_name ${ZC} --samples ${SAMPLES} --seed ${SEED} --arch ${ARCH} > ${SAVE_DIR}/rank_naive_zc_kd_sp_ps_${ZC}_${SAMPLES}_${SEED}.log 2>&1 &

# qe_score
ZC=qe_score
echo "Current zc: ${ZC}"
CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_zc_w_bit.py --data_path ${DATA_PATH} --zc_name ${ZC} --samples ${SAMPLES} --seed ${SEED} --arch ${ARCH} > ${SAVE_DIR}/rank_naive_zc_kd_sp_ps_${ZC}_${SAMPLES}_${SEED}.log 2>&1 &

# bparams
ZC=bparams
echo "Current zc: ${ZC}"
CUDA_VISIBLE_DEVICES=1 python exps/rank/rank_mqbench/rank_zc_w_bit.py --data_path ${DATA_PATH} --zc_name ${ZC} --samples ${SAMPLES} --seed ${SEED} --arch ${ARCH} > ${SAVE_DIR}/rank_naive_zc_kd_sp_ps_${ZC}_${SAMPLES}_${SEED}.log 2>&1 &

# bsynflow
ZC=bsynflow
echo "Current zc: ${ZC}"
CUDA_VISIBLE_DEVICES=0 python exps/rank/rank_mqbench/rank_zc_w_bit.py --data_path ${DATA_PATH} --zc_name ${ZC} --samples ${SAMPLES} --seed ${SEED} --arch ${ARCH} > ${SAVE_DIR}/rank_naive_zc_kd_sp_ps_${ZC}_${SAMPLES}_${SEED}.log 2>&1 &

# bsnip
ZC=bsnip
echo "Current zc: ${ZC}"
CUDA_VISIBLE_DEVICES=1 python exps/rank/rank_mqbench/rank_zc_w_bit.py --data_path ${DATA_PATH} --zc_name ${ZC} --samples ${SAMPLES} --seed ${SEED} --arch ${ARCH} > ${SAVE_DIR}/rank_naive_zc_kd_sp_ps_${ZC}_${SAMPLES}_${SEED}.log 2>&1 &


# echo total time
T="$(($(date +%s)-T))"
echo "It took ${T} seconds!"
