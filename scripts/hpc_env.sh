#!/bin/bash
# Source me in every shell session on Arnes HPC.
#   source scripts/hpc_env.sh
#
# Pins the toolchain to foss/2023a + CUDA 12.1.1 so the renderer build,
# vcpkg deps, and the PyTorch/2.1.2-foss-2023a-CUDA-12.1.1 module all
# share the same GCC/glibc/CUDA ABI.

module purge
module load foss/2023a
module load CUDA/12.1.1
module load CMake/3.26.3-GCCcore-12.3.0
module load Mesa/23.1.4-GCCcore-12.3.0

# The foss/2023a module injects CVMFS header paths into every GCC compile via
# CPATH and friends. That clobbers vcpkg's own bundled headers (zlib in
# particular — libpng then fails with PNG_ZLIB_VERNUM != ZLIB_VERNUM).
# Strip the include-path injection. We keep PATH / LD_LIBRARY_PATH / LIBRARY_PATH
# so the compiler binaries and link search still work normally.
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH

# Sanity print so we notice mismatches early.
echo "[hpc_env] gcc:   $(gcc --version | head -1)"
echo "[hpc_env] cmake: $(cmake --version | head -1)"
echo "[hpc_env] cuda:  $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not in PATH (ok on login node)')"
