"""Self-contained PSNR and SSIM (no skimage dependency, runs on any torch device).

Inputs are float tensors in [0,1], shaped [N,3,H,W] or [3,H,W].
"""

import torch
import torch.nn.functional as F


def _ensure_batched(x):
    return x.unsqueeze(0) if x.dim() == 3 else x


def psnr(pred, target, eps=1e-8):
    pred = _ensure_batched(pred).clamp(0, 1)
    target = _ensure_batched(target).clamp(0, 1)
    mse = F.mse_loss(pred, target, reduction='none').mean(dim=[1, 2, 3])
    return (10.0 * torch.log10(1.0 / (mse + eps))).mean().item()


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    win2d = (g.t() @ g).unsqueeze(0).unsqueeze(0)  # [1,1,W,W]
    return win2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim_index(pred, target, window_size=11, sigma=1.5):
    """Differentiable mean SSIM, returned as a **tensor** so it can be used
    inside a loss (e.g. L1 + lambda*(1 - ssim_index))."""
    pred = _ensure_batched(pred).clamp(0, 1)
    target = _ensure_batched(target).clamp(0, 1)
    c = pred.shape[1]
    win = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    pad = window_size // 2

    mu1 = F.conv2d(pred, win, padding=pad, groups=c)
    mu2 = F.conv2d(target, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu12 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu12

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


def ssim(pred, target, window_size=11, sigma=1.5):
    """Mean SSIM as a python float (for logging / metrics)."""
    return ssim_index(pred, target, window_size, sigma).item()
