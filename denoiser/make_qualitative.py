"""Build the qualitative comparison figure (the 'money shot') for the report:
a grid of crops, columns = methods, rows = frames, with PSNR/SSIM labels.

    noisy | Gaussian | cross-bilateral | KPCN (ours) | reference

Run on the machine that has the checkpoint + captured data (Mac or HPC):

    python denoiser/make_qualitative.py \
        --ckpt results/denoiser_counter-7k_kpcn/best.pt \
        --data data/renders --holdout counter-7k \
        --frames 2 --crop 180,160,140 \
        --out results/figures/fig_qualitative_counter.png

--crop x,y,size zooms into a square region (so edge-preservation is visible);
omit it to show full 512x512 frames. Baselines use the train-tuned params that
produced the report numbers (see FIG_PARAMS). PSNR/SSIM shown are FULL-FRAME for
that frame (matches the tables), regardless of the displayed crop.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from baselines import apply_method, psnr as np_psnr, ssim as np_ssim
from data_utils import (
    _load_depth, _load_noisy, _load_rgb, _normalize_depth,
    discover_samples, holdout_split, split_samples,
)
from model import UNetDenoiser
from train import pick_device

# Train-tuned baseline params (the winners from `baselines.py --tune`, identical
# optimum on counter and garden). Keep in sync if the grid search changes.
FIG_PARAMS = {
    'gauss': {'sigma': 1.5},
    'bilateral': {'sigma_s': 4.0, 'sigma_r': 0.1, 'guide_sigma': 1.5},
    'xbilat': {'sigma_s': 4.0, 'sigma_r': 0.1, 'sigma_d': 0.2, 'guide_sigma': 1.5},
}
COL_TITLES = {
    'noisy': 'noisy (1 spp)', 'gauss': 'Gaussian',
    'bilateral': 'bilateral', 'xbilat': 'cross-bilateral',
    'kpcn': 'KPCN (ours)', 'clean': 'reference (~400 spp)',
}


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt.get('args', {})
    model = UNetDenoiser(in_ch=4, out_ch=3, base=a.get('base', 32),
                         head=a.get('head', 'residual')).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, a.get('head', 'residual')


@torch.no_grad()
def _denoise(model, noisy, depth, device):
    inp = np.concatenate([noisy, depth[..., None]], axis=2)
    t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
    out = model(t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return out


def _crop(img, spec):
    if spec is None:
        return img
    x, y, s = spec
    return img[y:y + s, x:x + s]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='data/renders')
    ap.add_argument('--holdout', default=None,
                    help='show frames from this held-out scene (else pose split test).')
    ap.add_argument('--methods', default='noisy,gauss,xbilat,kpcn,clean',
                    help='columns, left->right (subset of '
                         'noisy,gauss,bilateral,xbilat,kpcn,clean).')
    ap.add_argument('--frames', type=int, default=2, help='# rows (test frames).')
    ap.add_argument('--indices', default=None,
                    help='comma-separated test-frame indices (overrides --frames).')
    ap.add_argument('--level', type=int, default=1, help='input spp (default 1=worst).')
    ap.add_argument('--crop', default=None,
                    help='x,y,size square zoom region (omit = full frame).')
    ap.add_argument('--out', default='results/figures/fig_qualitative.png')
    args = ap.parse_args()

    device = pick_device()
    model, head = _load_model(args.ckpt, device)
    methods = args.methods.split(',')
    crop = tuple(int(v) for v in args.crop.split(',')) if args.crop else None

    samples = discover_samples(args.data, [args.holdout] if args.holdout else None) \
        if args.holdout else discover_samples(args.data, None)
    if args.holdout:
        _, _, test_s = holdout_split(samples, args.holdout)
    else:
        _, _, test_s = split_samples(samples)
    if not test_s:
        raise SystemExit('no test frames found')

    if args.indices:
        idxs = [int(i) for i in args.indices.split(',')]
    else:
        # evenly spread across the test set for variety
        idxs = list(np.linspace(0, len(test_s) - 1, args.frames).round().astype(int))
    rows = [test_s[i] for i in idxs]
    print(f'ckpt head={head}, device={device}, {len(rows)} frame(s): indices {idxs}')

    nrow, ncol = len(rows), len(methods)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.2 * ncol, 2.5 * nrow),
                             squeeze=False)

    for r, (scene, stem, base, levels) in enumerate(rows):
        level = args.level if args.level in levels else min(levels)
        noisy = _load_noisy(base, level)
        clean = _load_rgb(f'{base}_clean.png')
        depth = _normalize_depth(_load_depth(f'{base}_depth.f32'))
        kpcn = _denoise(model, noisy, depth, device)

        imgs = {'noisy': noisy, 'clean': clean, 'kpcn': kpcn}
        for m in ('gauss', 'bilateral', 'xbilat'):
            if m in methods:
                imgs[m] = apply_method(m, noisy, depth, FIG_PARAMS[m])

        for c, m in enumerate(methods):
            ax = axes[r][c]
            ax.imshow(np.clip(_crop(imgs[m], crop), 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(COL_TITLES.get(m, m), fontsize=10,
                             fontweight='bold' if m == 'kpcn' else 'normal')
            if m not in ('clean',):  # full-frame metric vs reference
                p = np_psnr(imgs[m], clean)
                s = np_ssim(imgs[m], clean)
                ax.set_xlabel(f'{p:.2f} dB / {s:.3f}', fontsize=8)
        axes[r][0].set_ylabel(f'{scene}\n{stem}', fontsize=8)

    fig.suptitle('Qualitative comparison (metrics = full-frame PSNR/SSIM vs reference)',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
