#!/bin/bash
# Orchestrator: build the Apptainer container (one time) and the renderer.
#
# Workflow:
#   srun --partition=all --time=2:00:00 --cpus-per-task=8 --mem=16G --pty bash
#   bash scripts/hpc_build.sh
#
# Step 1 clones StochasticSplats if missing.
# Step 2 builds the Apptainer container if missing (~10-20 min one-time).
# Step 3 builds the renderer binary inside the container (fast, repeatable).
#
# The container provides Ubuntu 22.04 + all build deps via apt; no vcpkg.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
RENDERER_DIR="$REPO_ROOT/renderer/stochasticsplats"
RENDERER_URL="https://github.com/ubc-vision/stochasticsplats.git"
SIF="$REPO_ROOT/containers/renderer.sif"

# 1. Clone renderer if missing. Repo is git-ignored at our top level
#    (.gitignore: renderer/) so each environment fetches its own copy.
if [ ! -d "$RENDERER_DIR/.git" ]; then
    echo ">>> [1/3] Cloning StochasticSplats into $RENDERER_DIR ..."
    mkdir -p "$REPO_ROOT/renderer"
    # Shallow but with submodules — vcpkg is no longer used, so the submodule
    # is wasted bandwidth, but we keep --recursive in case the upstream repo
    # ever moves real code into a submodule.
    git clone --depth 1 --recurse-submodules --shallow-submodules \
        "$RENDERER_URL" "$RENDERER_DIR"
else
    echo ">>> [1/3] Renderer already cloned at $RENDERER_DIR"
fi

# 2. Build container if missing.
if [ ! -f "$SIF" ]; then
    echo ">>> [2/3] Building Apptainer container (one-time, ~10-20 min) ..."
    bash "$REPO_ROOT/scripts/build_container.sh"
else
    echo ">>> [2/3] Container already built at $SIF"
fi

# 3. Build renderer inside container.
echo ">>> [3/3] Building renderer inside container ..."
apptainer exec \
    --bind "$REPO_ROOT:$REPO_ROOT" \
    --pwd "$REPO_ROOT" \
    "$SIF" \
    bash "$REPO_ROOT/scripts/build_renderer.sh"

BIN="$RENDERER_DIR/build/splatapult"
if [ -x "$BIN" ]; then
    echo
    echo ">>> SUCCESS: $BIN"
    file "$BIN"
else
    echo ">>> FAIL: $BIN missing" >&2
    exit 1
fi
