#!/bin/bash
# Build splatapult using system Ubuntu libs (no vcpkg).
# Intended to be invoked INSIDE the apptainer container, e.g.:
#   apptainer exec --bind $PWD:$PWD --pwd $PWD containers/renderer.sif \
#       bash scripts/build_renderer.sh
#
# It will also run on bare Ubuntu 22.04 with the same apt packages installed,
# which is handy for local development on a Linux box.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
RENDERER_DIR="$REPO_ROOT/renderer/stochasticsplats"
BUILD_DIR="$RENDERER_DIR/build"

if [ ! -d "$RENDERER_DIR" ]; then
    echo "ERROR: renderer not found at $RENDERER_DIR" >&2
    echo "Run scripts/hpc_build.sh first (it clones the upstream repo)." >&2
    exit 1
fi

# Apply our overlay (drops X11/OpenXR find_package, stubs xrbuddy).
# Idempotent.
echo ">>> Applying renderer overlay ..."
REPO_ROOT="$REPO_ROOT" bash "$REPO_ROOT/scripts/patch_renderer.sh"

mkdir -p "$BUILD_DIR"
echo ">>> Configuring CMake (system packages, no vcpkg) ..."
cmake -S "$RENDERER_DIR" -B "$BUILD_DIR" \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release

JOBS="${JOBS:-$(nproc)}"
echo ">>> Building with -j${JOBS} ..."
cmake --build "$BUILD_DIR" --config Release -- -j"$JOBS"

BIN="$BUILD_DIR/splatapult"
if [ -x "$BIN" ]; then
    echo ">>> SUCCESS: $BIN"
    ls -lh "$BIN"
else
    echo ">>> FAIL: binary not found at $BIN" >&2
    exit 1
fi
