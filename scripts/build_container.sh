#!/bin/bash
# Build the Apptainer container with all renderer build deps.
# Run on a compute node (apptainer build is heavy) — login nodes may refuse.
#
#   srun --partition=all --time=2:00:00 --cpus-per-task=4 --mem=8G --pty bash
#   bash scripts/build_container.sh
#
# Takes ~10-20 min the first time. The resulting .sif is ~1 GB and lives
# inside the repo at containers/renderer.sif (gitignored).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
DEF="$REPO_ROOT/containers/renderer.def"
SIF="$REPO_ROOT/containers/renderer.sif"

if ! command -v apptainer >/dev/null; then
    echo "ERROR: apptainer not in PATH." >&2
    echo "On Arnes it's at /usr/bin/apptainer — your shell may have lost the system PATH." >&2
    exit 1
fi
if [ ! -f "$DEF" ]; then
    echo "ERROR: definition file missing at $DEF" >&2
    exit 1
fi

if [ -f "$SIF" ] && [ "${FORCE_REBUILD:-0}" != "1" ]; then
    echo ">>> Container already exists at $SIF"
    echo ">>> Delete it or set FORCE_REBUILD=1 to rebuild."
    exit 0
fi

mkdir -p "$(dirname "$SIF")"
echo ">>> Building $SIF from $DEF ..."
echo ">>> (10-20 min on first run; layers are cached in ~/.apptainer for re-runs)"

apptainer build "$SIF" "$DEF"

echo ">>> Done. Container at $SIF"
ls -lh "$SIF"
