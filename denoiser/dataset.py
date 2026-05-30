"""Dataset for the stochastic-splat denoiser.

Each sample on disk is a quadruple written by the WebGPU capture harness
(webgpu-splatting-dithering-nrg/capture.js) under data/renders/<scene>/<id>_*:

    <id>_noisy.png   single-frame (1 spp) stochastic render  -> network input
    <id>_clean.png   N-frame accumulated render               -> target
    <id>_depth.f32   raw linear depth, row-major float32, 0 = background
    <id>_depth.png   8-bit depth preview (not used for training)

The network input is 4 channels: noisy RGB in [0,1] plus a per-image
normalized depth channel (an edge/structure cue). Albedo is intentionally
excluded -- it leaks the clean color and would let the model cheat.
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset

# Torch-free helpers live in data_utils so the classical baselines can reuse
# them without importing PyTorch. Re-exported here for backward compatibility.
from data_utils import (  # noqa: F401
    SIZE,
    _load_depth,
    _load_noisy,
    _load_rgb,
    _noisy_levels,
    _normalize_depth,
    discover_samples,
    split_samples,
)


class SplatDenoiseDataset(Dataset):
    """Yields (input[4,H,W], target[3,H,W]) tensors.

    Training mode pulls a random crop and applies flips/90-rotations; eval mode
    returns the full image so PSNR/SSIM are measured on complete frames.
    """

    def __init__(self, samples, crop=128, train=True, augment=True, fixed_level=None):
        self.samples = samples
        self.crop = crop
        self.train = train
        self.augment = augment and train
        # fixed_level: pin the noise level (e.g. 1 = worst case) for eval; if None
        # in train mode, a random available level is drawn each access.
        self.fixed_level = fixed_level

    def __len__(self):
        return len(self.samples)

    def _augment(self, inp, tgt):
        # inp: [4,H,W], tgt: [3,H,W]
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[2]); tgt = torch.flip(tgt, dims=[2])
        if random.random() < 0.5:
            inp = torch.flip(inp, dims=[1]); tgt = torch.flip(tgt, dims=[1])
        k = random.randint(0, 3)
        if k:
            inp = torch.rot90(inp, k, dims=[1, 2]); tgt = torch.rot90(tgt, k, dims=[1, 2])
        return inp, tgt

    def __getitem__(self, idx):
        _, _, base, levels = self.samples[idx]
        if self.fixed_level is not None and self.fixed_level in levels:
            level = self.fixed_level
        elif self.train:
            level = random.choice(levels)                      # noise-level augmentation
        else:
            level = min(levels)                                # worst case for eval
        noisy = _load_noisy(base, level)                       # HxWx3
        clean = _load_rgb(f'{base}_clean.png')                 # HxWx3
        depth = _normalize_depth(_load_depth(f'{base}_depth.f32'))  # HxW

        inp = np.concatenate([noisy, depth[..., None]], axis=2)  # HxWx4
        inp = torch.from_numpy(inp).permute(2, 0, 1).contiguous()
        tgt = torch.from_numpy(clean).permute(2, 0, 1).contiguous()

        if self.train and self.crop and self.crop < inp.shape[1]:
            h, w = inp.shape[1], inp.shape[2]
            top = random.randint(0, h - self.crop)
            left = random.randint(0, w - self.crop)
            inp = inp[:, top:top + self.crop, left:left + self.crop]
            tgt = tgt[:, top:top + self.crop, left:left + self.crop]

        if self.augment:
            inp, tgt = self._augment(inp, tgt)

        return inp, tgt
