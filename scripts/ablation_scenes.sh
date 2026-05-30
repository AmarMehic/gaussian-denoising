#!/bin/bash
#SBATCH --job-name=splat-ablation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=results/ablation/slurm-%j.out

# Data-scaling ablation: does adding more *scenes* help generalization?
#
# The held-out TEST scene is fixed for every run; only the NUMBER of training
# scenes grows (4 -> 5 -> 6 -> 7). If held-out PSNR is still climbing at 7
# scenes, more data (scenes) is the fix; if it's flat, the bottleneck is the
# model/loss, not data. One figure that answers "would more data help?".
#
# Submit from the repo root (after `git pull` on HPC):
#     sbatch scripts/ablation_scenes.sh                 # counter-7k held out
#     sbatch scripts/ablation_scenes.sh garden-7k       # different held-out scene
#     sbatch scripts/ablation_scenes.sh counter-7k 0.0  # pure-L1 variant
# PREREQUISITE: `bash scripts/setup_env.sh` ONCE on the login node first.
set -e

module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source "$HOME/denoise-venv/bin/activate"

HOLDOUT="${1:-counter-7k}"   # fixed held-out TEST scene for EVERY run
SSIM_W="${2:-0.2}"           # match the main training config by default
EPOCHS="${3:-100}"

# Ordered pool of candidate TRAINING scenes (must NOT contain the holdout).
# Each ablation step appends the next scene, so diversity grows monotonically.
# Order alternates outdoor/indoor so even the small-N runs see varied content.
ALL=(bicycle-7k bonsai-7k garden-7k kitchen-7k stump-7k playroom-7k room-7k counter-7k)
POOL=()
for s in "${ALL[@]}"; do [ "$s" != "$HOLDOUT" ] && POOL+=("$s"); done

mkdir -p results/ablation
SUMMARY="results/ablation/summary_${HOLDOUT}.txt"
echo "data-scaling ablation | held-out test=$HOLDOUT | ssim_weight=$SSIM_W | epochs=$EPOCHS" | tee "$SUMMARY"
printf '%-8s %9s %9s %9s  %s\n' "n_train" "val_psnr" "test_psnr" "test_ssim" "train_scenes" | tee -a "$SUMMARY"

for N in 4 5 6 7; do
  TRAIN_SCENES=("${POOL[@]:0:$N}")
  OUT="results/ablation/${HOLDOUT}_n${N}"
  mkdir -p "$OUT"
  echo ">>> N=$N  train on: ${TRAIN_SCENES[*]}  | holdout $HOLDOUT"

  # --scenes restricts discovery to these N training scenes PLUS the holdout,
  # so holdout_split() carves the holdout out as the test set and splits the
  # remaining N scenes into train/val.
  python denoiser/train.py \
    --data data/renders \
    --scenes "${TRAIN_SCENES[@]}" "$HOLDOUT" \
    --holdout "$HOLDOUT" \
    --ssim_weight "$SSIM_W" \
    --epochs "$EPOCHS" --batch 16 --lr 1e-4 --crop 128 --base 32 --workers 8 \
    --out "$OUT" 2>&1 | tee "$OUT/train.txt"

  python denoiser/evaluate.py --ckpt "$OUT/best.pt" --data data/renders 2>&1 | tee "$OUT/eval.txt"

  # Scrape the headline numbers into the running summary table.
  VAL=$(grep -Eo 'best val PSNR [0-9.]+' "$OUT/train.txt" | grep -Eo '[0-9.]+' | tail -1)
  TLINE=$(grep -E '^denoised' "$OUT/eval.txt")
  TP=$(echo "$TLINE" | grep -Eo 'PSNR[[:space:]]+[0-9.]+' | grep -Eo '[0-9.]+')
  TS=$(echo "$TLINE" | grep -Eo 'SSIM[[:space:]]+[0-9.]+' | grep -Eo '[0-9.]+')
  printf '%-8s %9s %9s %9s  %s\n' "$N" "$VAL" "$TP" "$TS" "${TRAIN_SCENES[*]}" | tee -a "$SUMMARY"
done

echo ""
echo "=== ablation summary ($HOLDOUT held out) ==="
cat "$SUMMARY"
echo "trend: if test_psnr keeps rising 4->7, more scenes will help; if flat, data is not the bottleneck."
