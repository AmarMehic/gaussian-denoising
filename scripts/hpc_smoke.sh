#!/bin/bash
# Smoke test: does the renderer run headlessly inside the container?
#
# Submit on a GPU node:
#   srun --partition=gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=4 --mem=16G --pty bash
#   bash scripts/hpc_smoke.sh

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
BIN="$REPO_ROOT/renderer/stochasticsplats/build/splatapult"
PLY="$REPO_ROOT/data/scenes/bonsai/point_cloud/iteration_30000/point_cloud.ply"
SIF="$REPO_ROOT/containers/renderer.sif"

if [ ! -f "$SIF" ]; then
    echo "FAIL: container missing at $SIF — run scripts/hpc_build.sh first." >&2
    exit 1
fi
if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN missing — run scripts/hpc_build.sh first." >&2
    exit 1
fi
if [ ! -f "$PLY" ]; then
    # The plan.md path. Some scene downloads put the ply at scene root instead;
    # try the fallback location used by the HuggingFace dataset.
    ALT_PLY="$REPO_ROOT/data/scenes/bonsai/point_cloud.ply"
    if [ -f "$ALT_PLY" ]; then
        PLY="$ALT_PLY"
    else
        echo "FAIL: bonsai .ply not found at $PLY or $ALT_PLY" >&2
        exit 1
    fi
fi

echo ">>> Host GPU:"
nvidia-smi -L || echo "(no GPU visible — are you on a gpu partition node?)"
echo

run_with_driver() {
    local drv="$1"
    echo "----- SDL_VIDEODRIVER=$drv -----"
    SDL_VIDEODRIVER="$drv" timeout 8 \
        apptainer exec --nv \
            --cleanenv \
            --env SDL_VIDEODRIVER="$drv" \
            --bind "$REPO_ROOT:$REPO_ROOT" \
            --pwd "$REPO_ROOT/renderer/stochasticsplats" \
            "$SIF" \
            "$BIN" "$PLY" 2>&1 | tail -30
    local rc=${PIPESTATUS[0]}
    # timeout returns 124 when it kills the process — that means the renderer
    # was running happily when we cut it off. Success for this test.
    if [ "$rc" = "124" ] || [ "$rc" = "0" ]; then
        echo "----- $drv: OK (rc=$rc) -----"
        return 0
    else
        echo "----- $drv: FAIL (rc=$rc) -----"
        return 1
    fi
}

# offscreen: SDL2's headless backend (no display). Built into SDL2 — works.
# dummy:    even lower-fi backend; useful diagnostic but no OpenGL context.
# x11:      requires a real display; included only for completeness.
for drv in offscreen x11 dummy; do
    if run_with_driver "$drv"; then
        echo
        echo ">>> WINNER: SDL_VIDEODRIVER=$drv runs without a display."
        echo ">>> Use this driver in dataset generation scripts."
        exit 0
    fi
done

echo
echo ">>> No SDL video driver worked. Next: try Xvfb inside the container."
echo "    apptainer exec --nv $SIF xvfb-run -a $BIN $PLY"
exit 1
