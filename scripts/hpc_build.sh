#!/bin/bash
# Build the (unmodified) StochasticSplats renderer on Arnes HPC.
#
# Assumes scripts/hpc_env.sh has been sourced in this shell.
# Run from the repo root:
#   source scripts/hpc_env.sh
#   bash scripts/hpc_build.sh
#
# vcpkg builds many dependencies from source the first time — this can
# take 20-40 min. Prefer running on a compute node, e.g.:
#   srun --partition=all --time=2:00:00 --cpus-per-task=8 --mem=16G --pty bash
#   source scripts/hpc_env.sh
#   bash scripts/hpc_build.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
RENDERER_DIR="$REPO_ROOT/renderer/stochasticsplats"
RENDERER_URL="https://github.com/ubc-vision/stochasticsplats.git"

if ! command -v cmake >/dev/null; then
    echo "ERROR: cmake not found. Did you 'source scripts/hpc_env.sh'?" >&2
    exit 1
fi

# 1. Clone renderer if missing. The repo is excluded from our git tree
#    (.gitignore: renderer/) so each environment fetches its own copy.
if [ ! -d "$RENDERER_DIR/.git" ]; then
    echo ">>> Cloning StochasticSplats into $RENDERER_DIR ..."
    mkdir -p "$REPO_ROOT/renderer"
    git clone --recursive "$RENDERER_URL" "$RENDERER_DIR"
else
    echo ">>> Renderer already cloned; updating submodules"
    git -C "$RENDERER_DIR" submodule update --init --recursive
fi

# 1b. Apply our headless overlay (removes X11/OpenXR dependencies).
#     Idempotent — safe across re-runs.
echo ">>> Applying renderer overlay (headless: no X11, no OpenXR) ..."
REPO_ROOT="$REPO_ROOT" bash "$REPO_ROOT/scripts/patch_renderer.sh"

# 2. Bootstrap vcpkg (submodule)
VCPKG_DIR="$RENDERER_DIR/vcpkg"
VCPKG_BIN="$VCPKG_DIR/vcpkg"
if [ ! -x "$VCPKG_BIN" ]; then
    echo ">>> Bootstrapping vcpkg ..."
    (cd "$VCPKG_DIR" && ./bootstrap-vcpkg.sh -disableMetrics)
fi

# 3. CMake configure + build
BUILD_DIR="$RENDERER_DIR/build"
mkdir -p "$BUILD_DIR"
echo ">>> Configuring CMake (toolchain via vcpkg) ..."
cmake -S "$RENDERER_DIR" -B "$BUILD_DIR" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$VCPKG_DIR/scripts/buildsystems/vcpkg.cmake"

JOBS="${JOBS:-$(nproc)}"
echo ">>> Building with -j${JOBS} ..."
cmake --build "$BUILD_DIR" --config Release -- -j"$JOBS"

BIN="$BUILD_DIR/splatapult"
if [ -x "$BIN" ]; then
    echo ">>> SUCCESS: binary at $BIN"
    file "$BIN"
else
    echo ">>> FAIL: binary not found at $BIN" >&2
    exit 1
fi
