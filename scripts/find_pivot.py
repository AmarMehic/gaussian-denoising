#!/usr/bin/env python3
"""Find a good orbit pivot for a .splat scene, in the SAME centered coordinate
frame the capture harness uses (webgpu-splatting-dithering-nrg/capture.js).

Steps (mirroring capture.js):
  1. Robust trimmed centroid of all splat positions: start at the mean, then
     iteratively keep only the inner 60% and recompute (4 passes). This is the
     scene-centering origin -- background walls/ground don't drag it off.
  2. Center positions on that origin.
  3. Find the densest cluster (the subject): voxelize the centered cloud, take
     the densest voxel, then grow a ball around it a few times to settle on the
     cluster centroid. Report a percentile radius -> a suggested camera distance.

Prints a ready-to-paste SCENE_PRESETS line. Verify/tune dist in-browser.

    python3 scripts/find_pivot.py data/scenes/garden-7k.splat [more.splat ...]
"""

import sys
from pathlib import Path

import numpy as np


def load_positions(path):
    raw = np.fromfile(path, dtype=np.uint8)
    n = raw.size // 32
    raw = raw[: n * 32].reshape(n, 32)
    # position = first 12 bytes reinterpreted as 3 float32
    pos = raw[:, :12].copy().view(np.float32).reshape(n, 3)
    return pos.astype(np.float64)


def robust_center(pos, passes=4, keep=0.6):
    c = pos.mean(axis=0)
    for _ in range(passes):
        d = np.linalg.norm(pos - c, axis=1)
        cut = np.quantile(d, keep)
        inner = pos[d <= cut]
        if len(inner):
            c = inner.mean(axis=0)
    return c


def densest_cluster(centered, voxel=0.5, grow_radius=2.0, grow_passes=4):
    # Densest voxel as a seed.
    keys = np.floor(centered / voxel).astype(np.int64)
    # hash voxel coords to find the most populated cell
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    seed_cell = uniq[counts.argmax()]
    seed = (seed_cell + 0.5) * voxel

    # Grow a ball around the seed, re-centering on the enclosed mean.
    c = seed.astype(np.float64)
    for _ in range(grow_passes):
        d = np.linalg.norm(centered - c, axis=1)
        ball = centered[d <= grow_radius]
        if len(ball) < 50:
            break
        c = ball.mean(axis=0)

    d = np.linalg.norm(centered - c, axis=1)
    in_ball = d[d <= grow_radius]
    p85 = float(np.quantile(in_ball, 0.85)) if len(in_ball) else grow_radius
    frac = float((d <= grow_radius).mean())
    return c, p85, frac


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    for path in sys.argv[1:]:
        name = Path(path).stem
        pos = load_positions(path)
        center = robust_center(pos)
        centered = pos - center
        c, p85, frac = densest_cluster(centered)
        dist = round(max(2.4 * p85, 3.0), 1)  # initial guess; tune in-browser
        tx, ty, tz = (round(float(v), 2) for v in c)
        print(f"    '{name}': {{ tx: {tx}, ty: {ty}, tz: {tz}, dist: {dist} }},"
              f"  // cluster p85 r={p85:.2f}, {frac*100:.0f}% of splats in ball, "
              f"{len(pos):,} splats")


if __name__ == '__main__':
    main()
