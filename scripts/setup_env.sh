#!/bin/bash
# One-time training-environment setup for Arnes. Run ONCE on the LOGIN NODE
# (it needs internet for a single small pip install; compute nodes are often
# air-gapped, so this must not happen inside the SLURM job):
#
#     bash scripts/setup_env.sh
#
# Strategy: the cluster's PyTorch EasyBuild module already provides torch +
# numpy (and CUDA). It does NOT provide Pillow, which our PNG I/O needs. So we
# layer a tiny venv on top with --system-site-packages (inherits torch/numpy
# from the module, no multi-GB reinstall) and add only Pillow. The training job
# (scripts/slurm_train.sh) then just `module load` + activates this venv.
set -e

MODULE="PyTorch/2.1.2-foss-2023a-CUDA-12.1.1"   # matches requirements: torch>=2.1, has CUDA
VENV="$HOME/denoise-venv"

module purge
module load "$MODULE"
echo "loaded $MODULE"

if [ ! -d "$VENV" ]; then
  echo "creating venv at $VENV (inherits the module's torch + numpy)"
  python -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"

# Pillow is the only dependency the module is missing.
python -c "import PIL" 2>/dev/null || pip install --upgrade Pillow

echo "=== sanity check ==="
python -c "import torch, numpy, PIL; print('torch', torch.__version__, '| numpy', numpy.__version__, '| Pillow', PIL.__version__, '| cuda_build', torch.version.cuda, '| cuda_avail', torch.cuda.is_available())"
echo
echo "env ready. Submit training with:  sbatch scripts/slurm_train.sh"
echo "(cuda_avail shows False on the login node -- that's expected; it is True on a GPU compute node.)"
