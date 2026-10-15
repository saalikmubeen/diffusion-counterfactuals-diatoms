#!/usr/bin/env bash
# Fine-tune the 256x256 unconditional diffusion model on DIATLAS train+val.
# One MPI process per GPU. Extra arguments are passed to train_diffusion.py.
#
#   GPUS=0,1 RUN=ft_v1 bash diatoms/train_diffusion.sh
#   GPUS=0 RUN=probe bash diatoms/train_diffusion.sh --total_steps 50 --save_interval 1000000
#
# Checkpoints and logs (log.txt, progress.csv) go to checkpoints/diatlas/$RUN.
# To resume: --resume_checkpoint checkpoints/diatlas/$RUN/modelNNNNNN.pt
# (the optimizer and EMA files next to it are picked up automatically).
set -euo pipefail
cd "$(dirname "$0")/.."

GPUS=${GPUS:-0,1}
RUN=${RUN:-ft_$(date +%Y%m%d_%H%M)}
ENV_BIN=${ENV_BIN:-$HOME/miniconda3/envs/python_38_dvces_cu111/bin}
PY=$ENV_BIN/python
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")

export OPENAI_LOGDIR=checkpoints/diatlas/$RUN
# The two RTX 3090s on this machine don't support direct GPU-to-GPU (P2P)
# access ("peer access is not supported"), so NCCL must go through host memory.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
mkdir -p "$OPENAI_LOGDIR"
echo "run=$RUN gpus=$GPUS nproc=$NPROC logdir=$OPENAI_LOGDIR"

CUDA_VISIBLE_DEVICES=$GPUS "$ENV_BIN/mpiexec" -n "$NPROC" --bind-to none -x CUDA_VISIBLE_DEVICES -x OPENAI_LOGDIR -x NCCL_P2P_DISABLE \
    "$PY" diatoms/train_diffusion.py "$@"
