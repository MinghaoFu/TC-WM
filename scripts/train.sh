#!/bin/bash
# Train a world model on a single task.
# Choose GPUs by exporting CUDA_VISIBLE_DEVICES before launching.
#
#   CONFIG_NAME=train_tcwm TASK=wall bash scripts/train.sh
#   CUDA_VISIBLE_DEVICES=0 TASK=lift bash scripts/train.sh

set -e
cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-offline}
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

CONFIG_NAME="${CONFIG_NAME:-train_tcwm}"
TASK="${TASK:-wall}"
EPOCHS="${EPOCHS:-100}"

python train.py --config-name="$CONFIG_NAME" env="$TASK" training.epochs="$EPOCHS"
