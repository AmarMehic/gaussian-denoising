#!/bin/bash
#SBATCH --job-name=splat-denoise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=results/denoiser/slurm-%j.out

# Arnes HPC training job.  Submit from the repo root:  sbatch scripts/slurm_train.sh
# PREREQUISITE: run `bash scripts/setup_env.sh` ONCE on the login node first
# (it creates the $HOME/denoise-venv this job activates).
set -e

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$HOME/denoise-venv/bin/activate"

# Fail fast with a clear message if the environment isn't set up.
python -c "import torch, numpy, PIL; print('env ok: torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo 'ENV BROKEN -- run `bash scripts/setup_env.sh` on the login node first'; exit 1; }

mkdir -p results/denoiser

# Leave-one-scene-out: train on 7 scenes, test on the held-out one (unseen
# geometry/content -> a true generalization number for the report).
HOLDOUT=counter-7k

python denoiser/train.py \
  --data data/renders \
  --holdout "$HOLDOUT" \
  --epochs 100 \
  --batch 16 \
  --lr 1e-4 \
  --crop 128 \
  --base 32 \
  --workers 8 \
  --out results/denoiser

# evaluate.py reads the held-out scene from the checkpoint automatically.
python denoiser/evaluate.py --ckpt results/denoiser/best.pt --data data/renders

# Classical baseline on the SAME held-out scene (apples-to-apples comparison).
python denoiser/baselines.py --data data/renders --holdout "$HOLDOUT" --split test
