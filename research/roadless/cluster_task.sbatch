#!/bin/bash
# One array task of a cluster.py run: line SLURM_ARRAY_TASK_ID + 1 of the task list ($1) is a
# script and its arguments, run in the reblock pixi env with the block bank ($2).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
set -euo pipefail
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")
cd "$HOME/src/reblock"
# shellcheck source=cluster_env.sh
source research/roadless/cluster_env.sh
export REBLOCK_BLOCK_BANK=$2
echo "task ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname), commit $(cat .cluster_commit): $LINE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
# shellcheck disable=SC2086  # LINE is a script and its arguments
"$HOME/.pixi/bin/pixi" run --frozen python -u $LINE
