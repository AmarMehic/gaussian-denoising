"""Evaluate a trained denoiser: PSNR/SSIM on the held-out test split,
the noisy-input baseline for reference, and inference latency per frame.

    python denoiser/evaluate.py --ckpt results/denoiser/best.pt --data data/renders
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import SplatDenoiseDataset, discover_samples, split_samples
from metrics import psnr, ssim
from model import UNetDenoiser
from train import pick_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/denoiser/best.pt')
    ap.add_argument('--data', default='data/renders')
    ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    base = ckpt.get('args', {}).get('base', 32)
    model = UNetDenoiser(in_ch=4, out_ch=3, base=base).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'device: {device}, checkpoint: {args.ckpt} (base={base})')

    samples = discover_samples(args.data, args.scenes)
    _, _, test_s = split_samples(samples, seed=args.seed)
    test_dl = DataLoader(SplatDenoiseDataset(test_s, train=False), batch_size=1)
    print(f'test frames: {len(test_s)}')

    den_p = den_s = base_p = base_s = 0.0
    latencies = []
    for inp, tgt in test_dl:
        inp, tgt = inp.to(device), tgt.to(device)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(inp).clamp(0, 1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        latencies.append((time.time() - t0) * 1000)

        noisy_rgb = inp[:, :3]
        den_p += psnr(out, tgt); den_s += ssim(out, tgt)
        base_p += psnr(noisy_rgb, tgt); base_s += ssim(noisy_rgb, tgt)

    n = len(test_s)
    # First call includes lazy kernel compilation; report the warm median.
    warm = sorted(latencies[1:] or latencies)
    median_ms = warm[len(warm) // 2]

    print('\n=== Test results (mean over {} frames) ==='.format(n))
    print(f'noisy  -> clean : PSNR {base_p/n:6.2f} dB   SSIM {base_s/n:.4f}   (baseline)')
    print(f'denoised        : PSNR {den_p/n:6.2f} dB   SSIM {den_s/n:.4f}')
    print(f'improvement     : +{(den_p-base_p)/n:.2f} dB   +{(den_s-base_s)/n:.4f}')
    print(f'latency / 512x512 frame: {median_ms:.1f} ms (median, warm) on {device}')


if __name__ == '__main__':
    main()
