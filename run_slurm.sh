#!/bin/bash

module load anaconda
module load cuda/11.1
module load cudnn/8.1.0.77_CUDA11.1
module load gcc/7.5
source activate py38

bash scripts/search/emq_zc/run_evo_search_emq_zc.sh
