#!/bin/bash
# Probe the Arnes (or any SLURM cluster) software stack so we know exactly how to
# load PyTorch in the training job. Run this on the LOGIN NODE and paste the
# output back:
#
#     bash scripts/check_env.sh
#
# It is read-only: it only lists modules and tries imports, changes nothing.

echo "=== host ==="
hostname
echo

echo "=== relevant modules (terse) ==="
# -t = terse (one per line); merge stderr because module prints there.
module -t avail 2>&1 | grep -iE 'pytorch|python|cuda|cudnn|foss|anaconda|conda' | sort -u
echo

echo "=== default python ==="
which python python3 2>&1
python3 --version 2>&1
echo

echo "=== torch in the bare environment? ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda_build', torch.version.cuda)" 2>&1
echo

echo "=== existing conda/venv hints ==="
echo "CONDA_PREFIX=${CONDA_PREFIX:-<none>}"
echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<none>}"
ls -d "$HOME"/*env* "$HOME"/miniconda* "$HOME"/anaconda* 2>/dev/null || echo "(no obvious env dirs in \$HOME)"
echo

echo "=== done. Paste everything above back. ==="
