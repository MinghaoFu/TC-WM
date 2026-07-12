#!/bin/bash
# Run TC-WM training on one or more tasks.
# Choose GPUs by exporting CUDA_VISIBLE_DEVICES before launching.
#
#   TASKS=lift bash scripts/run_tcwm.sh                 # single task, foreground
#   TASKS=lift,can,square bash scripts/run_tcwm.sh      # several tasks, background (one log each)

set -euo pipefail
cd "$(dirname "$0")/.."

export WANDB_MODE=${WANDB_MODE:-offline}
EPOCHS="${EPOCHS:-100}"

IFS=',' read -r -a TASKS <<< "${TASKS:-lift}"

if [ "${#TASKS[@]}" -eq 1 ]; then
  python train.py --config-name=train_tcwm env="${TASKS[0]}" training.epochs="$EPOCHS"
else
  mkdir -p logs
  for task in "${TASKS[@]}"; do
    echo "Launching $task -> logs/${task}.log"
    nohup python train.py --config-name=train_tcwm env="$task" training.epochs="$EPOCHS" > "logs/${task}.log" 2>&1 &
  done
  echo "Launched ${#TASKS[@]} tasks. Pin GPUs per process with CUDA_VISIBLE_DEVICES."
  wait
fi
