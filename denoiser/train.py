"""Train the U-Net denoiser on captured stochastic-splat renders.

Example (HPC):
    python denoiser/train.py --data data/renders --epochs 100 --batch 16 --lr 1e-4

Auto-selects CUDA (HPC), then MPS (Apple Silicon), then CPU.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SplatDenoiseDataset, discover_samples, split_samples
from metrics import psnr, ssim
from model import UNetDenoiser


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ss, n = 0.0, 0.0, 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        out = model(inp).clamp(0, 1)
        ps += psnr(out, tgt) * inp.size(0)
        ss += ssim(out, tgt) * inp.size(0)
        n += inp.size(0)
    return ps / max(n, 1), ss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/renders', help='renders root (scene subdirs)')
    ap.add_argument('--scenes', nargs='*', default=None, help='limit to these scenes')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--base', type=int, default=32, help='U-Net base channel width')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', default='results/denoiser', help='checkpoint/log dir')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'device: {device}')

    samples = discover_samples(args.data, args.scenes)
    if not samples:
        raise SystemExit(f'No samples found under {args.data}. Run the capture first.')
    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    print(f'samples: {len(samples)} total -> {len(train_s)} train / {len(val_s)} val / {len(test_s)} test')

    train_ds = SplatDenoiseDataset(train_s, crop=args.crop, train=True)
    val_ds = SplatDenoiseDataset(val_s, train=False)

    pin = device.type == 'cuda'
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=pin, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=pin)

    model = UNetDenoiser(in_ch=4, out_ch=3, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.L1Loss()
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'model: UNetDenoiser base={args.base}, {n_params/1e6:.2f}M params')

    best_psnr = -1.0
    log_path = out_dir / 'train_log.csv'
    log_path.write_text('epoch,train_loss,val_psnr,val_ssim,lr,sec\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for inp, tgt in train_dl:
            inp, tgt = inp.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', enabled=use_amp):
                out = model(inp)
                loss = loss_fn(out, tgt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * inp.size(0)
        sched.step()

        train_loss = running / len(train_ds)
        val_psnr, val_ssim = evaluate(model, val_dl, device)
        dt = time.time() - t0
        lr = opt.param_groups[0]['lr']
        print(f'[{epoch:3d}/{args.epochs}] loss {train_loss:.4f}  '
              f'val PSNR {val_psnr:.2f} dB  SSIM {val_ssim:.4f}  ({dt:.1f}s)')
        with log_path.open('a') as f:
            f.write(f'{epoch},{train_loss:.6f},{val_psnr:.4f},{val_ssim:.4f},{lr:.2e},{dt:.1f}\n')

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({'model': model.state_dict(), 'args': vars(args),
                        'epoch': epoch, 'val_psnr': val_psnr},
                       out_dir / 'best.pt')

    torch.save({'model': model.state_dict(), 'args': vars(args)}, out_dir / 'last.pt')
    print(f'done. best val PSNR {best_psnr:.2f} dB -> {out_dir/"best.pt"}')


if __name__ == '__main__':
    main()
