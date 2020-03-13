#!/bin/bash

# Copyright  2018  Northwestern Polytechnical University (author: Ke Wang)

set -euo pipefail

lr="1e-3"
data_dir="data"                   #mirana's dir
dconv_norm_type='gLN'
active_func="relu"
date=$(date "+%Y%m%d")
causal="false"
save_name="tasnet_${date}_${active_func}_${dconv_norm_type}_${lr}"
mkdir -p exp/${save_name}

num_gpu=6                                          #mirana's gpu
batch_size_single_gpu=4
batch_size=$[num_gpu*batch_size_single_gpu]
CUDA_VISIBLE_DEVICES='2,3,4,5,6,7' python -u steps/run_tasnet.py \
    --decode="false" \
    --batch-size=${batch_size} \
    --learning-rate=${lr} \
    --weight-decay=1e-5 \
    --epochs=100 \
    --data-dir=${data_dir} \
    --modelDir="exp/${save_name}" \
    --use-cuda="true" \
    --autoencoder-channels=256 \
    --autoencoder-kernel-size=20 \
    --bottleneck-channels=256 \
    --convolution-channels=512 \
    --convolution-kernel-size=3 \
    --number-blocks=8 \
    --number-repeat=4 \
    --number-speakers=2 \
    --normalization-type=${dconv_norm_type} \
    --active-func=${active_func} \
    --causal=${causal}
