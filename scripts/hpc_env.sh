#!/bin/bash
# Source me in shells where you want to run the PyTorch / denoiser side
# natively on Arnes HPC. The renderer side does NOT need this — it lives
# inside the Apptainer container at containers/renderer.sif.
#
#   source scripts/hpc_env.sh
#
# Toolchain pinned to foss/2023a + CUDA 12.1.1 to align with the
# PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 module we'll use for training.

module purge
module load foss/2023a
module load CUDA/12.1.1
module load CMake/3.26.3-GCCcore-12.3.0

# Don't let the module system inject /cvmfs/... include paths into every
# native compile — those would override container-provided headers if you
# ever do a hybrid build. Harmless to unset even when only doing PyTorch.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH

echo "[hpc_env] gcc:   $(gcc --version | head -1)"
echo "[hpc_env] cmake: $(cmake --version | head -1)"
echo "[hpc_env] cuda:  $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH (ok on login node)')"
