"""Evaluate a trained denoiser: PSNR/SSIM on the held-out test split,
the noisy-input baseline for reference, and inference latency per frame.

    python denoiser/evaluate.py --ckpt results/denoiser/best.pt --data data/renders
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import (
    SplatDenoiseDataset,
    discover_samples,
    holdout_split,
    split_samples,
)
from metrics import psnr, ssim
from model import UNetDenoiser
from train import pick_device


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/denoiser/best.pt')
    ap.add_argument('--data', default='data/renders')
    ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--holdout', default=None,
                    help='override the held-out test scene; defaults to whatever '
                         'the checkpoint was trained with.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--level', type=int, default=None,
                    help='pin the input noise level (spp): 1, 2 or 4. Default '
                         'None = worst case (1 spp / min available per sample).')
    ap.add_argument('--half', action='store_true',
                    help='run the conv body in fp16 (CUDA/MPS) for a free latency '
                         'win. KPCN softmax/unfold stays fp32, so quality is ~unchanged.')
    ap.add_argument('--scale', type=float, default=1.0,
                    help='denoise at this fraction of native res (e.g. 0.5 = 256px) '
                         'for a latency/quality tradeoff; output is upsampled back to '
                         'full res for PSNR/SSIM. Latency is timed at the reduced res.')
    args = ap.parse_args()

    device = pick_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    ck_args = ckpt.get('args', {})
    base = ck_args.get('base', 32)
    head = ck_args.get('head', 'residual')  # old checkpoints predate the flag
    # Mirror training's split so we evaluate on the exact same held-out test set.
    holdout = args.holdout or ck_args.get('holdout')
    seed = ck_args.get('seed', args.seed)
    model = UNetDenoiser(in_ch=4, out_ch=3, base=base, head=head).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    if args.half and device.type in ('cuda', 'mps'):
        model = model.half()
    print(f'device: {device}, checkpoint: {args.ckpt} (base={base}, head={head}'
          f'{", fp16" if args.half else ""})')

    samples = discover_samples(args.data, args.scenes)
    if holdout:
        _, _, test_s = holdout_split(samples, holdout, seed=seed)
        print(f'held-out test scene: {holdout!r}')
    else:
        _, _, test_s = split_samples(samples, seed=seed)
    test_dl = DataLoader(
        SplatDenoiseDataset(test_s, train=False, fixed_level=args.level), batch_size=1)
    lvl = f'{args.level} spp' if args.level else 'worst case (1 spp)'
    print(f'test frames: {len(test_s)}  |  input noise level: {lvl}')

    den_p = den_s = base_p = base_s = 0.0
    latencies = []
    # Per-scene accumulators: scene -> [den_psnr, den_ssim, base_psnr, base_ssim, n].
    # batch_size=1 + shuffle=False, so the i-th batch aligns with test_s[i].
    per_scene = {}
    for i, (inp, tgt) in enumerate(test_dl):
        inp, tgt = inp.to(device), tgt.to(device)
        full_hw = inp.shape[-2:]
        # Feed the model fp16 when --half; keep `inp`/`tgt` fp32 so metrics below
        # are computed in full precision (the cast is cheap and outside the timer).
        inp_run = inp.half() if args.half else inp
        # Optionally denoise at reduced resolution: shrink the 4-ch input here
        # (outside the timer), time the forward at that res, then upsample the
        # 3-ch output back to native res so PSNR/SSIM compare against the full tgt.
        if args.scale != 1.0:
            inp_run = F.interpolate(inp_run, scale_factor=args.scale,
                                    mode='bilinear', align_corners=False)

        # Both CUDA and MPS dispatch GPU work asynchronously, so we must block
        # until the device is idle on BOTH sides of the timed region -- otherwise
        # time.time() stops before the GPU finishes and the latency reads far too
        # low (especially on MPS, which has no implicit sync here).
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        t0 = time.time()
        out = model(inp_run).clamp(0, 1)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()
        latencies.append((time.time() - t0) * 1000)
        proc_hw = tuple(inp_run.shape[-2:])  # resolution the network actually ran at

        out = out.float()
        if out.shape[-2:] != full_hw:
            out = F.interpolate(out, size=full_hw, mode='bilinear',
                                align_corners=False).clamp(0, 1)
        noisy_rgb = inp[:, :3]
        dp, ds = psnr(out, tgt), ssim(out, tgt)
        bp, bs = psnr(noisy_rgb, tgt), ssim(noisy_rgb, tgt)
        den_p += dp; den_s += ds; base_p += bp; base_s += bs

        scene = test_s[i][0]
        acc = per_scene.setdefault(scene, [0.0, 0.0, 0.0, 0.0, 0])
        acc[0] += dp; acc[1] += ds; acc[2] += bp; acc[3] += bs; acc[4] += 1

    n = len(test_s)
    # First call includes lazy kernel compilation; report the warm median.
    warm = sorted(latencies[1:] or latencies)
    median_ms = warm[len(warm) // 2]

    print('\n=== Test results (mean over {} frames) ==='.format(n))
    print(f'noisy  -> clean : PSNR {base_p/n:6.2f} dB   SSIM {base_s/n:.4f}   (baseline)')
    print(f'denoised        : PSNR {den_p/n:6.2f} dB   SSIM {den_s/n:.4f}')
    print(f'improvement     : +{(den_p-base_p)/n:.2f} dB   +{(den_s-base_s)/n:.4f}')

    if len(per_scene) > 1:
        print('\n--- per-scene (denoised PSNR/SSIM | noisy PSNR/SSIM) ---')
        for scene in sorted(per_scene):
            dp, ds, bp, bs, k = per_scene[scene]
            print(f'{scene:<14} {dp/k:6.2f} {ds/k:.4f}  |  {bp/k:6.2f} {bs/k:.4f}  ({k})')
    ph, pw = proc_hw
    note = f' (denoised at {pw}x{ph}, upsampled to {full_hw[1]}x{full_hw[0]})' \
        if args.scale != 1.0 else ''
    print(f'latency / {pw}x{ph} frame: {median_ms:.1f} ms (median, warm) on {device}{note}')


if __name__ == '__main__':
    main()
