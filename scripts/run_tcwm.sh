#!/bin/bash

# Unified TCWM runner
# - Single task: default foreground (debug) run
# - Multiple tasks: default background (nohup) run
# - DEBUG=true forces foreground; DEBUG=false forces background
#
# Usage:
#   TASKS=lift scripts/run_tcwm.sh
#   TASKS=lift,pusht scripts/run_tcwm.sh
#   DEBUG=false TASKS=lift scripts/run_tcwm.sh
#   DEBUG=true TASKS=lift,pusht scripts/run_tcwm.sh

set -euo pipefail

cd "$(dirname "$0")/.."

PID_FILE="logs/run_tcwm.pids"
JOB_FILE="logs/run_tcwm.jobs"
GPU_IDS="${GPU_IDS:-auto}"
TASKS_RAW="${TASKS:-}"

DEFAULT_TASKS=(
  lift
  # can
  # square
  # pusht
)

if [ -n "$TASKS_RAW" ]; then
  IFS=',' read -r -a TASKS <<< "$TASKS_RAW"
else
  TASKS=("${DEFAULT_TASKS[@]}")
fi

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "❌ No tasks specified. Set TASKS=task1,task2 or edit DEFAULT_TASKS in run_tcwm.sh."
  exit 1
fi

select_gpus() {
  if [ "$GPU_IDS" = "auto" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        sort -k2 -n | awk -F',' '{print $1}' | tr -d ' '
    fi
  else
    echo "$GPU_IDS" | tr ',' '\n'
  fi
}

mapfile -t AVAILABLE_GPUS < <(select_gpus)
if [ "${#AVAILABLE_GPUS[@]}" -eq 0 ]; then
  echo "No GPUs detected; tasks will run without CUDA_VISIBLE_DEVICES."
else
  echo "Available GPUs: ${AVAILABLE_GPUS[*]}"
fi

mkdir -p logs
echo "Launched at $(date)" > "$JOB_FILE"
> "$PID_FILE"

num_tasks="${#TASKS[@]}"
debug_flag="${DEBUG:-}"

if [ -z "$debug_flag" ]; then
  if [ "$num_tasks" -eq 1 ]; then
    debug_flag="true"
  else
    debug_flag="false"
  fi
fi

run_foreground() {
  local task="$1"
  local gpu="$2"
  echo "Starting task (foreground): $task"
  if [ -n "$gpu" ]; then
    echo "  GPU: $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=offline python train.py --config-name=train_tcwm env="$task" training.epochs="${EPOCHS:-100}" debug=true
  else
    WANDB_MODE=offline python train.py --config-name=train_tcwm env="$task" training.epochs="${EPOCHS:-100}" debug=true
  fi
}

run_background() {
  local task="$1"
  local gpu="$2"
  local log_file="logs/${task}.log"
  echo "Starting task (background): $task"
  if [ -n "$gpu" ]; then
    echo "  GPU: $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=offline nohup python train.py --config-name=train_tcwm env="$task" training.epochs="${EPOCHS:-100}" > "$log_file" 2>&1 &
  else
    WANDB_MODE=offline nohup python train.py --config-name=train_tcwm env="$task" training.epochs="${EPOCHS:-100}" > "$log_file" 2>&1 &
  fi
  local pid=$!
  echo "$pid" >> "$PID_FILE"
  printf "%s\t%s\t%s\n" "$pid" "$task" "$log_file" >> "$JOB_FILE"
  echo "  PID: $pid, log: $log_file"
}

gpu_idx=0
if [ "$debug_flag" = "true" ]; then
  # Foreground (serial)
  for task in "${TASKS[@]}"; do
    gpu=""
    if [ "${#AVAILABLE_GPUS[@]}" -gt 0 ]; then
      gpu="${AVAILABLE_GPUS[$gpu_idx]}"
      gpu_idx=$(( (gpu_idx + 1) % ${#AVAILABLE_GPUS[@]} ))
    fi
    run_foreground "$task" "$gpu"
  done
else
  # Background (parallel)
  for task in "${TASKS[@]}"; do
    gpu=""
    if [ "${#AVAILABLE_GPUS[@]}" -gt 0 ]; then
      gpu="${AVAILABLE_GPUS[$gpu_idx]}"
      gpu_idx=$(( (gpu_idx + 1) % ${#AVAILABLE_GPUS[@]} ))
    fi
    run_background "$task" "$gpu"
  done
  echo "PIDs saved to: $PID_FILE"
  echo "Jobs saved to: $JOB_FILE"
fi
