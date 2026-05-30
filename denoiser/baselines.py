"""Training-free classical baselines for the stochastic-splat denoiser.

These give us a reference point *before* the U-Net is trained, so we can confirm
the capture/eval pipeline works end to end and know how much a learned model has
to beat. Everything here is pure numpy + Pillow (no torch / scipy / cv2 / skimage)
so it runs on a laptop on the frames already on disk.

Baselines, weakest to strongest:
  noisy     - the 1-spp input itself (the floor every method must beat)
  gauss     - a fixed Gaussian blur (kills noise *and* detail; edge-blind)
  bilateral - color-only bilateral filter (edge-aware on RGB)
  xbilat    - depth-guided cross-bilateral filter (Mara 2017 aligned): the same
              spatial + range weighting, but the range term also includes the
              depth channel, so it stops smoothing across depth discontinuities.

Run on whatever is on disk (defaults to the test split, worst-case noise level):

    python denoiser/baselines.py --data data/renders
    python denoiser/baselines.py --data data/renders --scenes bonsai-7k \
        --save results/baselines/bonsai
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from data_utils import (
    _load_depth,
    _load_noisy,
    _load_rgb,
    _normalize_depth,
    discover_samples,
    holdout_split,
    split_samples,
)


# ----------------------------------------------------------------------------
# Metrics (numpy; mirror denoiser/metrics.py which is torch-only)
# ----------------------------------------------------------------------------

def psnr(a, b):
    """PSNR in dB between two [0,1] HxWx3 arrays."""
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def _gauss_kernel1d(sigma, radius):
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _sep_blur(img, sigma):
    """Separable Gaussian blur of an HxWx C array (reflect padding)."""
    if sigma <= 0:
        return img
    radius = max(1, int(round(3 * sigma)))
    k = _gauss_kernel1d(sigma, radius).astype(np.float32)
    out = img.astype(np.float32)
    # horizontal
    pad = np.pad(out, ((0, 0), (radius, radius), (0, 0)), mode='reflect')
    acc = np.zeros_like(out)
    for i, w in enumerate(k):
        acc += w * pad[:, i:i + out.shape[1], :]
    out = acc
    # vertical
    pad = np.pad(out, ((radius, radius), (0, 0), (0, 0)), mode='reflect')
    acc = np.zeros_like(out)
    for i, w in enumerate(k):
        acc += w * pad[i:i + out.shape[0], :, :]
    return acc


def ssim(a, b, sigma=1.5):
    """Mean SSIM over RGB between two [0,1] HxWx3 arrays (Gaussian windows)."""
    C1, C2 = (0.01 ** 2), (0.03 ** 2)
    mu_a = _sep_blur(a, sigma)
    mu_b = _sep_blur(b, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = _sep_blur(a * a, sigma) - mu_a2
    sb = _sep_blur(b * b, sigma) - mu_b2
    sab = _sep_blur(a * b, sigma) - mu_ab
    num = (2 * mu_ab + C1) * (2 * sab + C2)
    den = (mu_a2 + mu_b2 + C1) * (sa + sb + C2)
    return float(np.mean(num / den))


# ----------------------------------------------------------------------------
# Filters (vectorized shifted-window accumulation -> no python pixel loops)
# ----------------------------------------------------------------------------

def gaussian_baseline(noisy, sigma=1.2):
    return np.clip(_sep_blur(noisy, sigma), 0.0, 1.0)


def _shift(arr, dy, dx):
    """Shift an HxW[xC] array by (dy,dx) with reflect padding."""
    r = max(abs(dy), abs(dx))
    if arr.ndim == 2:
        pad = np.pad(arr, ((r, r), (r, r)), mode='reflect')
    else:
        pad = np.pad(arr, ((r, r), (r, r), (0, 0)), mode='reflect')
    h, w = arr.shape[:2]
    return pad[r + dy:r + dy + h, r + dx:r + dx + w]


def _bilateral(noisy, depth=None, radius=5, sigma_s=3.0, sigma_r=0.25,
               sigma_d=0.15, guide_sigma=1.5):
    """Edge-aware bilateral / cross-bilateral filter.

    Spatial weight from pixel offset (sigma_s); range weight from color
    difference (sigma_r). At 1 spp the raw input is so noisy that per-pixel
    color differences are dominated by noise, so the range term is computed
    against a pre-smoothed *guide* image (joint-bilateral style) -- otherwise
    the filter mistakes noise for edges and preserves it. If `depth` is given,
    the range weight also includes the normalized depth difference (sigma_d),
    giving a depth-guided cross-bilateral filter that stops smoothing across
    depth discontinuities (Mara 2017 style).
    """
    h, w, _ = noisy.shape
    guide = _sep_blur(noisy, guide_sigma)  # stable edge/range reference
    out = np.zeros_like(noisy, dtype=np.float32)
    wsum = np.zeros((h, w, 1), dtype=np.float32)
    inv_2ss = 1.0 / (2.0 * sigma_s ** 2)
    inv_2sr = 1.0 / (2.0 * sigma_r ** 2)
    inv_2sd = 1.0 / (2.0 * sigma_d ** 2)
    cdepth = depth[..., None] if depth is not None else None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = np.exp(-(dy * dy + dx * dx) * inv_2ss)
            nb = _shift(noisy, dy, dx)
            gdiff = np.sum((_shift(guide, dy, dx) - guide) ** 2, axis=2, keepdims=True)
            wgt = ws * np.exp(-gdiff * inv_2sr)
            if cdepth is not None:
                nd = _shift(depth, dy, dx)[..., None]
                ddiff = (nd - cdepth) ** 2
                wgt = wgt * np.exp(-ddiff * inv_2sd)
            out += wgt * nb
            wsum += wgt
    return np.clip(out / np.maximum(wsum, 1e-8), 0.0, 1.0)


def bilateral_baseline(noisy, **kw):
    return _bilateral(noisy, depth=None, **kw)


def xbilateral_baseline(noisy, depth, **kw):
    return _bilateral(noisy, depth=depth, **kw)


# ----------------------------------------------------------------------------
# Methods + (optionally tuned) hyperparameters
# ----------------------------------------------------------------------------

METHOD_NAMES = ['noisy', 'gauss', 'bilateral', 'xbilat']

# Hand-picked defaults = the "untuned" baseline.
DEFAULT_PARAMS = {
    'noisy': {},
    'gauss': {'sigma': 1.2},
    'bilateral': {},   # _bilateral() signature defaults
    'xbilat': {},
}

# Grids searched by --tune. Kept modest: the bilateral filters cost ~1s/img, so
# (#configs x #tune images) is the time budget. radius is left at its default.
TUNE_GRIDS = {
    'gauss': [{'sigma': s} for s in (0.8, 1.0, 1.2, 1.5, 2.0)],
    'bilateral': [
        {'sigma_s': ss, 'sigma_r': sr, 'guide_sigma': gs}
        for ss in (2.0, 3.0, 4.0)
        for sr in (0.10, 0.15, 0.20, 0.30)
        for gs in (1.0, 1.5)
    ],
    'xbilat': [
        {'sigma_s': ss, 'sigma_r': sr, 'sigma_d': sd, 'guide_sigma': 1.5}
        for ss in (2.0, 3.0, 4.0)
        for sr in (0.10, 0.15, 0.20, 0.30)
        for sd in (0.05, 0.10, 0.15, 0.20)
    ],
}


def apply_method(m, noisy, depth, params):
    if m == 'noisy':
        return noisy
    if m == 'gauss':
        return gaussian_baseline(noisy, **params)
    if m == 'bilateral':
        return bilateral_baseline(noisy, **params)
    if m == 'xbilat':
        return xbilateral_baseline(noisy, depth, **params)
    raise ValueError(f'unknown method {m!r}')


# ----------------------------------------------------------------------------
# Tuning (grid-search on the TRAIN split, freeze for the held-out test eval)
# ----------------------------------------------------------------------------

def _spread_subset(samples, limit, seed=0):
    """Pick up to `limit` samples spread round-robin across scenes (stable)."""
    by_scene = {}
    for s in samples:
        by_scene.setdefault(s[0], []).append(s)
    rng = random.Random(seed)
    pools = {sc: rng.sample(v, len(v)) for sc, v in by_scene.items()}
    scenes = sorted(pools)
    picks, i = [], 0
    while len(picks) < limit and any(pools.values()):
        sc = scenes[i % len(scenes)]
        if pools[sc]:
            picks.append(pools[sc].pop())
        i += 1
    return picks


def _load_triple(sample, level_arg):
    _, _, base, levels = sample
    level = level_arg if (level_arg in levels) else min(levels)
    noisy = _load_noisy(base, level)
    clean = _load_rgb(f'{base}_clean.png')
    depth = _normalize_depth(_load_depth(f'{base}_depth.f32'))
    return noisy, clean, depth


def tune_on_train(train_samples, level_arg, metric='psnr', limit=12, seed=0):
    """Grid-search each tunable filter on a TRAIN subset; return chosen params.

    FAIRNESS: tuning only ever sees TRAIN scenes (the same data the U-Net learns
    from). The selected hyperparameters are then frozen and evaluated on the
    held-out test scene -- never tuned on the test scene. This mirrors the
    learned model's train/test protocol for an honest comparison.
    """
    pick = _spread_subset(train_samples, limit, seed=seed)
    loaded = [_load_triple(s, level_arg) for s in pick]
    score_fn = psnr if metric == 'psnr' else ssim
    chosen = dict(DEFAULT_PARAMS)
    n_scenes = len({s[0] for s in pick})
    print(f'tuning on {len(loaded)} train images over {n_scenes} scenes '
          f'(objective: {metric})')
    for m in ('gauss', 'bilateral', 'xbilat'):
        grid = TUNE_GRIDS[m]
        best, best_score = None, -1e9
        for params in grid:
            sc = sum(score_fn(apply_method(m, n, d, params), c)
                     for n, c, d in loaded) / len(loaded)
            if sc > best_score:
                best_score, best = sc, params
        chosen[m] = best
        print(f'  {m:<10} best train {metric} {best_score:6.3f}  ->  {best}  '
              f'({len(grid)} configs)')
    return chosen


# ----------------------------------------------------------------------------
# Eval driver
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/renders')
    ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--split', default='test', choices=['test', 'val', 'all'])
    ap.add_argument('--holdout', default=None,
                    help='evaluate on this held-out scene (match the U-Net run '
                         'for an apples-to-apples baseline).')
    ap.add_argument('--level', type=int, default=None,
                    help='noise spp level (default: worst available per sample)')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap number of samples (quick smoke test)')
    ap.add_argument('--save', default=None,
                    help='dir to write side-by-side comparison PNGs')
    ap.add_argument('--tune', action='store_true',
                    help='grid-search filter hyperparameters on the TRAIN split, '
                         'freeze, then evaluate on the held-out test split '
                         '(fair, apples-to-apples with the learned model).')
    ap.add_argument('--tune_limit', type=int, default=12,
                    help='# train images used for tuning (bilateral ~1s/img).')
    ap.add_argument('--tune_metric', default='psnr', choices=['psnr', 'ssim'],
                    help='objective the grid search maximizes on the train split.')
    args = ap.parse_args()

    samples = discover_samples(args.data, args.scenes)
    if not samples:
        raise SystemExit(f'no samples found under {args.data}')
    if args.split == 'all':
        train_s, subset = samples, samples
    elif args.holdout:
        train_s, val, test = holdout_split(samples, args.holdout)
        subset = test if args.split == 'test' else val
    else:
        train_s, val, test = split_samples(samples)
        subset = test if args.split == 'test' else val
    if args.limit:
        subset = subset[:args.limit]

    if args.tune:
        params = tune_on_train(train_s, args.level, args.tune_metric, args.tune_limit)
        tag = f'TUNED on train (objective {args.tune_metric})'
    else:
        params = dict(DEFAULT_PARAMS)
        tag = 'default (untuned)'

    print(f'{len(subset)} samples ({args.split} split) from {args.data}  |  params: {tag}')
    print(f'{"method":<10} {"PSNR(dB)":>9} {"SSIM":>7} {"sec/img":>8}')

    sums = {m: [0.0, 0.0, 0.0] for m in METHOD_NAMES}  # psnr, ssim, time
    # per_scene[scene][method] = [psnr, ssim, n]
    per_scene = {}
    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for scene, stem, base, levels in subset:
        level = args.level if (args.level in levels) else min(levels)
        noisy = _load_noisy(base, level)
        clean = _load_rgb(f'{base}_clean.png')
        depth = _normalize_depth(_load_depth(f'{base}_depth.f32'))

        sc = per_scene.setdefault(scene, {m: [0.0, 0.0, 0] for m in METHOD_NAMES})
        panels = []
        for m in METHOD_NAMES:
            t0 = time.perf_counter()
            out = apply_method(m, noisy, depth, params[m])
            dt = time.perf_counter() - t0
            p, s = psnr(out, clean), ssim(out, clean)
            sums[m][0] += p; sums[m][1] += s; sums[m][2] += dt
            sc[m][0] += p; sc[m][1] += s; sc[m][2] += 1
            if save_dir:
                panels.append(out)

        if save_dir:
            strip = np.concatenate(panels + [clean], axis=1)
            Image.fromarray((strip * 255).round().clip(0, 255).astype(np.uint8)).save(
                save_dir / f'{scene}_{stem}_baselines.png')

    n = len(subset)
    for m in METHOD_NAMES:
        p, s, t = (v / n for v in sums[m])
        print(f'{m:<10} {p:>9.2f} {s:>7.4f} {t:>8.3f}')

    if len(per_scene) > 1:
        print('\n--- per-scene PSNR(dB) / SSIM by method ---')
        for scene in sorted(per_scene):
            print(f'[{scene}]')
            for m in METHOD_NAMES:
                p, s, k = per_scene[scene][m]
                print(f'  {m:<10} {p/k:>9.2f} {s/k:>7.4f}  ({k})')

    if args.tune:
        print('\nfrozen params (selected on train, applied to test):')
        for m in ('gauss', 'bilateral', 'xbilat'):
            print(f'  {m:<10} {params[m]}')

    if save_dir:
        print(f'wrote comparison strips (noisy|gauss|bilateral|xbilat|clean) to {save_dir}')


if __name__ == '__main__':
    main()
