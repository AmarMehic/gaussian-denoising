#!/bin/bash
#SBATCH --job-name=splat-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=results/denoiser_smoke/slurm-%j.out

# Quick end-to-end SMOKE TEST: 2 epochs into a separate output dir so it can't
# touch the real run's checkpoints. The short 30-min walltime backfills onto a
# GPU node fast. It exercises the whole path -- module/venv, dataset discovery,
# the held-out split, the dataloader (PNG + depth decode), the train loop
# (forward/backward/AMP/scheduler), checkpoint save, and the eval + metrics +
# latency code. If this finishes clean, submit the full run:
#
#     sbatch scripts/slurm_train.sh
set -e

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$HOME/denoise-venv/bin/activate"

python -c "import torch, numpy, PIL; print('env ok: torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo 'ENV BROKEN -- run `bash scripts/setup_env.sh` on the login node first'; exit 1; }

mkdir -p results/denoiser_smoke

HOLDOUT=counter-7k

python denoiser/train.py \
  --data data/renders \
  --holdout "$HOLDOUT" \
  --epochs 2 \
  --batch 16 \
  --lr 1e-4 \
  --crop 128 \
  --base 32 \
  --workers 8 \
  --out results/denoiser_smoke

# Confirms checkpoint -> eval path works (the held-out scene is read from the ckpt).
python denoiser/evaluate.py --ckpt results/denoiser_smoke/best.pt --data data/renders

echo "SMOKE TEST OK -- submit the full run with: sbatch scripts/slurm_train.sh"
