#!/bin/bash
#SBATCH --job-name=splat-baselines
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/baselines/slurm-%j.out

# Tuned classical baselines on GPU, with the CPU/GPU parity check, for the
# apples-to-apples latency comparison vs. the learned model.
#
# Submit from the repo root (DETACHED -- you can log off / sleep):
#     mkdir -p results/baselines
#     sbatch scripts/slurm_baselines.sh
# Then read the log:
#     tail -f results/baselines/slurm-<jobid>.out
#
# PREREQUISITE: `bash scripts/setup_env.sh` was run once on the login node.
set -e

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$HOME/denoise-venv/bin/activate"

python -c "import torch; print('cuda available:', torch.cuda.is_available())"

for HOLDOUT in counter-7k garden-7k; do
  echo "===== ${HOLDOUT} ====="
  python denoiser/baselines.py --data data/renders --holdout "$HOLDOUT" \
      --split test --tune --device auto --parity
done

echo "ALL DONE"
