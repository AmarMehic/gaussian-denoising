#!/bin/bash
#SBATCH --job-name=splat-denoise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=results/denoiser/slurm-%j.out

# Arnes HPC training job.  Submit from the repo root:
#     sbatch scripts/slurm_train.sh              # defaults to counter-7k
#     sbatch scripts/slurm_train.sh garden-7k    # hold out a different scene
# Each held-out scene writes to its own results/denoiser_<scene> dir, so you can
# run several in parallel (one GPU each, ~10 min) without clobbering.
# PREREQUISITE: run `bash scripts/setup_env.sh` ONCE on the login node first.
set -e

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$HOME/denoise-venv/bin/activate"

# Fail fast with a clear message if the environment isn't set up.
python -c "import torch, numpy, PIL; print('env ok: torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || { echo 'ENV BROKEN -- run `bash scripts/setup_env.sh` on the login node first'; exit 1; }

# Leave-one-scene-out: train on 7 scenes, test on the held-out one (unseen
# geometry/content -> a true generalization number for the report).
HOLDOUT="${1:-counter-7k}"        # first sbatch arg, default counter-7k
SSIM_W="${2:-0.0}"                # weight on (1-SSIM) loss term; 0 = pure L1.
                                  # Held-out tests showed L1+0.2*SSIM was WORSE
                                  # than pure L1 (see notes.md S8), so default 0.
HEAD="${3:-residual}"             # output head: 'residual' (default) or 'kpcn'
                                  # (kernel-predicting, edge-preserving filter).
# Separate out dir per head so residual and kpcn runs don't clobber each other.
SUFFIX=""; [ "$HEAD" != "residual" ] && SUFFIX="_${HEAD}"
OUT="results/denoiser_${HOLDOUT}${SUFFIX}"
echo "holdout=$HOLDOUT  ssim_weight=$SSIM_W  head=$HEAD  out=$OUT"
mkdir -p "$OUT"

python denoiser/train.py \
  --data data/renders \
  --holdout "$HOLDOUT" \
  --ssim_weight "$SSIM_W" \
  --head "$HEAD" \
  --epochs 100 \
  --batch 16 \
  --lr 1e-4 \
  --crop 128 \
  --base 32 \
  --workers 8 \
  --out "$OUT"

# evaluate.py reads the held-out scene from the checkpoint automatically.
python denoiser/evaluate.py --ckpt "$OUT/best.pt" --data data/renders

# Classical baseline on the SAME held-out scene (apples-to-apples comparison).
python denoiser/baselines.py --data data/renders --holdout "$HOLDOUT" --split test
