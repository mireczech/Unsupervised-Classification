#!/bin/bash

# The actual script
DATASETS=("matek" "isic" "retinopathy" "jurkat" "cifar10")

for dataset in "${DATASETS[@]}"; do
    echo "running scan for '$dataset' dataset"

    python mireczech_scan.py \
        --config_env mireczech/configs/env.yml \
        --config_exp mireczech/configs/scan/scan_$dataset.yml > scan_$dataset.log
done
