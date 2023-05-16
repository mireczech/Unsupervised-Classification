#!/bin/bash

# Define default values
main_repo=""

# Print help message
function print_help {
    echo "Usage: $0 [OPTIONS]"
    echo "OPTIONS:"
    echo "    -r|--main-repo      Main repo path"
    echo "    -h|--help           Print this help message"
    exit 0
}

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -r|--main-repo)
            main_repo="$2"
            shift 2
            ;;
        -h|--help)
            print_help
            ;;
        *)
            echo "Invalid option: $1"
            print_help
            ;;
    esac
done

# Check mandatory arguments
if [ -z "$main_repo" ]; then
    echo "Error: Mandatory arguments not provided"
    print_help
fi

# Removing potential trailing slash
main_repo=${main_repo%/}

# Get the script's directory
script_dir=$(dirname "$(readlink -f "$0")")

# Resolve the relative path
main_repo=$(readlink -f "$main_repo")

echo "Data source: $main_repo"

# The actual script
DATASETS=("matek" "isic" "retinopathy" "jurkat" "cifar10")

declare -A simclr_weights
simclr_weights["matek"]="outputs/2023-04-06/12-19-01/lightning_logs/uq7vzdxh/checkpoints/epoch=984-step=27580-val_loss_ssl=5.32.ckpt"
simclr_weights["isic"]="outputs/2023-04-05/23-04-29/lightning_logs/rbn9afqg/checkpoints/epoch=999-step=39000-val_loss_ssl=5.33.ckpt"
simclr_weights["retinopathy"]="outputs/2023-04-05/18-49-26/lightning_logs/y52wl6b8/checkpoints/epoch=974-step=4875-val_loss_ssl=5.04.ckpt"
simclr_weights["jurkat"]="outputs/2023-04-10/14-51-10/lightning_logs/vw8jsukl/checkpoints/epoch=999-step=12000-val_loss_ssl=6.57.ckpt"
simclr_weights["cifar10"]="outputs/2023-04-09/18-03-54/lightning_logs/1lcvwh2w/checkpoints/epoch=989-step=11880-val_loss_ssl=7.58.ckpt"

mkdir -p mireczech/data

for dataset in "${DATASETS[@]}"; do
    echo "setting up everything for '$dataset' dataset"

    cp -rf $main_repo/src/datasets/data/$dataset mireczech/data
    mkdir -p mireczech/results/$dataset/pretext
    cp $main_repo/${simclr_weights[$dataset]} mireczech/results/$dataset/pretext/model.pth.tar
done
