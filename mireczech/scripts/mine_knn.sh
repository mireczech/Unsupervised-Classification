#!/bin/bash

# The actual script
DATASETS=("matek" "isic" "retinopathy" "jurkat" "cifar10")

for dataset in "${DATASETS[@]}"; do
    echo "mining knn for '$dataset' dataset"

    python mireczech_mine_knn.py \
        --config_env mireczech/configs/env.yml \
        --config_exp mireczech/configs/mine_knn/simclr_$dataset.yml

    echo -e "\n\n"
done
