#!/bin/bash
# Smoke test: does the unmodified renderer run without a display?
#
# Requires a GPU. Submit from the repo root via:
#   srun --partition=gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=4 --mem=16G --pty bash
#   source scripts/hpc_env.sh
#   bash scripts/hpc_smoke.sh
#
# We try a few SDL2 video drivers in order of preference and report which
# (if any) lets the renderer start without a connected display.

set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
BIN="$REPO_ROOT/renderer/stochasticsplats/build/splatapult"
PLY="$REPO_ROOT/data/scenes/bonsai/point_cloud/iteration_30000/point_cloud.ply"

if [ ! -x "$BIN" ]; then
    echo "FAIL: $BIN missing. Run scripts/hpc_build.sh first." >&2
    exit 1
fi
if [ ! -f "$PLY" ]; then
    echo "FAIL: bonsai .ply not found at $PLY" >&2
    exit 1
fi

echo ">>> nvidia-smi:"
nvidia-smi -L || echo "(no GPU visible — are you on a gpu partition node?)"
echo

run_with_driver() {
    local drv="$1"
    echo "----- Trying SDL_VIDEODRIVER=$drv -----"
    SDL_VIDEODRIVER="$drv" timeout 8 "$BIN" "$PLY" 2>&1 | tail -30
    local rc=${PIPESTATUS[0]}
    # timeout exits 124 when it kills the process — that means the renderer
    # was happily running when we cut it off, which is success for this test.
    if [ "$rc" = "124" ] || [ "$rc" = "0" ]; then
        echo "----- $drv: OK (rc=$rc) -----"
        return 0
    else
        echo "----- $drv: FAIL (rc=$rc) -----"
        return 1
    fi
}

# Order: offscreen is SDL2's built-in headless backend (no X needed).
# dummy is even lower-fi (no GL context); shouldn't work for us but useful
# diagnostic. x11 requires a display server — included only for completeness.
for drv in offscreen dummy x11; do
    if run_with_driver "$drv"; then
        echo
        echo ">>> WINNER: SDL_VIDEODRIVER=$drv runs without a display."
        echo ">>> Use 'export SDL_VIDEODRIVER=$drv' before running the renderer."
        exit 0
    fi
done

echo
echo ">>> No SDL video driver worked. Next step: port context creation to EGL,"
echo ">>> or build inside an apptainer container that bundles Xvfb."
exit 1
