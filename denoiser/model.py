"""U-Net denoiser: 4-channel input (noisy RGB + depth) -> 3-channel clean RGB.

A standard encoder/decoder U-Net with skip connections. Two output heads:

  - **residual** (default): the final 1x1 conv predicts an RGB residual that is
    added to the noisy input. Usual denoising parameterization -- the model only
    learns the noise to remove, and the identity (no-op) is easy to represent.
    Downside: it can output *any* RGB value, so an L1 objective happily produces
    a blurry local average.

  - **kpcn** (kernel-predicting, Bako 2017 / Gharbi 2019): the final conv predicts
    a K*K *kernel* per pixel; the kernel is softmax-normalized and applied to the
    noisy RGB's KxK neighborhood (a per-pixel weighted average). Because the
    output is a normalized average of *real input pixels*, the network can only
    redistribute existing signal -- it cannot hallucinate a blurry smear -- so it
    is structurally edge-preserving (a learned, spatially-varying bilateral). This
    directly targets the L1-blur failure mode of the residual head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetDenoiser(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base=32, head='residual', kernel_size=11):
        super().__init__()
        assert head in ('residual', 'kpcn'), f'unknown head {head!r}'
        self.head_type = head
        self.out_ch = out_ch
        self.kernel_size = kernel_size

        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.enc4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        if head == 'residual':
            self.head = nn.Conv2d(base, out_ch, 1)
        else:  # kpcn: one K*K kernel per output pixel
            self.head = nn.Conv2d(base, kernel_size * kernel_size, 1)

    def _apply_kpcn(self, weights, noisy):
        """Apply a predicted per-pixel kernel to the noisy RGB.

        weights: [N, K*K, H, W] raw logits; noisy: [N, 3, H, W] in [0,1].
        Returns [N, 3, H, W] = per-pixel softmax-weighted average of the KxK
        neighborhood. Softmax is done in fp32 for stability under AMP.
        """
        n, _, h, w = noisy.shape
        k = self.kernel_size
        ks = k * k
        # Normalize the kernel so the output is a convex combination of pixels.
        w_soft = F.softmax(weights.float(), dim=1)              # [N, K*K, H, W]
        # KxK neighborhood of every pixel: [N, 3*K*K, H*W] -> [N, 3, K*K, H*W].
        patches = F.unfold(noisy.float(), kernel_size=k, padding=k // 2)
        patches = patches.view(n, self.out_ch, ks, h * w)
        w_soft = w_soft.view(n, 1, ks, h * w)                   # broadcast over RGB
        out = (patches * w_soft).sum(dim=2)                     # [N, 3, H*W]
        return out.view(n, self.out_ch, h, w)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        h = self.head(d1)
        if self.head_type == 'residual':
            return h + x[:, :3]            # add back the noisy RGB
        return self._apply_kpcn(h, x[:, :3])
