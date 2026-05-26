#!/bin/bash
# Apply our headless modifications to the cloned StochasticSplats renderer.
#
# The renderer lives at renderer/stochasticsplats/ (git-ignored). The overlay
# files at scripts/renderer_overlay/ shadow specific files in that tree. This
# script copies the overlay onto the cloned renderer.
#
# Changes:
#   - Remove openxr-loader from vcpkg.json (no VR, no OpenXR build)
#   - Remove find_package(X11)/find_package(OpenXR) from CMakeLists.txt
#   - Stub out xrbuddy.h/.cpp (so all the if(vrMode) code paths compile)
#   - Remove the X11/GLX SysWMinfo grab in sdl_main.cpp
#   - Strip X11/GLX fields from maincontext.h on Linux
#
# Idempotent: safe to re-run.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OVERLAY="$REPO_ROOT/scripts/renderer_overlay"
TARGET="$REPO_ROOT/renderer/stochasticsplats"

if [ ! -d "$TARGET" ]; then
    echo "ERROR: renderer not cloned at $TARGET — run hpc_build.sh first." >&2
    exit 1
fi
if [ ! -d "$OVERLAY" ]; then
    echo "ERROR: overlay not found at $OVERLAY — bad repo checkout?" >&2
    exit 1
fi

# Walk the overlay tree and copy each file to the matching path in the renderer.
# `find ... -type f` so we don't try to copy directories.
count=0
while IFS= read -r src; do
    rel="${src#$OVERLAY/}"
    dst="$TARGET/$rel"
    mkdir -p "$(dirname "$dst")"
    cp -v "$src" "$dst"
    count=$((count + 1))
done < <(find "$OVERLAY" -type f)

echo
echo ">>> Patched $count files into $TARGET"
