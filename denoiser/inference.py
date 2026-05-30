"""Run a trained denoiser on one captured frame and write a side-by-side PNG
(noisy | denoised | clean) plus the standalone denoised image.

    python denoiser/inference.py --ckpt results/denoiser/best.pt \
        --base data/renders/bonsai-7k/0000 --out results/denoiser/0000_compare.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from dataset import _load_depth, _load_noisy, _load_rgb, _normalize_depth, _noisy_levels
from model import UNetDenoiser
from train import pick_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/denoiser/best.pt')
    ap.add_argument('--base', required=True, help='sample path prefix, e.g. data/renders/bonsai-7k/0000')
    ap.add_argument('--level', type=int, default=1, help='noisy spp level to denoise (default 1 = worst)')
    ap.add_argument('--out', default=None, help='output comparison PNG path')
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    width = ckpt.get('args', {}).get('base', 32)
    model = UNetDenoiser(in_ch=4, out_ch=3, base=width).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    base = Path(args.base)
    levels = _noisy_levels(base.parent, base.name)
    level = args.level if args.level in levels else (levels[0] if levels else args.level)
    noisy = _load_noisy(args.base, level)
    depth = _normalize_depth(_load_depth(f'{args.base}_depth.f32'))
    inp = np.concatenate([noisy, depth[..., None]], axis=2)
    inp_t = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)

    out = model(inp_t).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()

    clean_path = f'{args.base}_clean.png'
    panels = [noisy, out]
    if Path(clean_path).exists():
        panels.append(_load_rgb(clean_path))
    strip = np.concatenate(panels, axis=1)
    strip_img = Image.fromarray((strip * 255).round().clip(0, 255).astype(np.uint8))

    out_path = Path(args.out) if args.out else Path(f'{args.base}_compare.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip_img.save(out_path)
    Image.fromarray((out * 255).round().clip(0, 255).astype(np.uint8)).save(
        out_path.with_name(out_path.stem + '_denoised.png'))
    print(f'wrote {out_path} (noisy | denoised{" | clean" if len(panels) == 3 else ""})')


if __name__ == '__main__':
    main()
