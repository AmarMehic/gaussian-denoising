"""Generate the data-driven report figures from the numbers recorded in notes.md
(Section 8). Self-contained: pure matplotlib, no torch / dataset / GPU needed, so
it runs anywhere (laptop, login node) in a second.

    python scripts/make_plots.py --out results/figures

Produces four PNGs (300 dpi):
    fig_head_swap.png       residual vs KPCN head (the single-variable proof)
    fig_data_scaling.png    test PSNR vs #train scenes (diminishing returns)
    fig_noise_levels.png    input vs denoised PSNR at 1/2/4 spp
    fig_quality_latency.png PSNR vs ms/frame (KPCN top-left, bilaterals far right)

If any number here drifts from notes.md, fix it HERE too -- this is the single
source the figures are rendered from.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless / no display (works over SSH)
import matplotlib.pyplot as plt

# Consistent palette so colors mean the same thing across all figures.
C_KPCN = '#1b6ca8'      # ours
C_RESID = '#9aa7b1'     # residual U-Net
C_CLASS = '#d1495b'     # classical filters
C_NOISY = '#b0b0b0'     # noisy input
C_ACCENT = '#e08e0b'


# ---------------------------------------------------------------------------
# Figure 4 in the report: head-swap bar chart (counter-7k, pure L1).
# The single-variable proof: same data/loss, only the output head changes.
# ---------------------------------------------------------------------------
def fig_head_swap(out_dir):
    labels = ['residual head', 'KPCN head (ours)']
    psnr = [24.64, 27.17]
    ssim = [0.615, 0.724]
    colors = [C_RESID, C_KPCN]

    fig, (axp, axs) = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, vals, title, fmt in (
            (axp, psnr, 'PSNR (dB)', '{:.2f}'),
            (axs, ssim, 'SSIM', '{:.3f}')):
        bars = ax.bar(labels, vals, color=colors, width=0.6,
                      edgecolor='black', linewidth=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(vals) * 1.18)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=0.25)
    fig.suptitle('Output head is the lever (counter-7k held out, same data & L1 loss)',
                 fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, 'fig_head_swap.png')


# ---------------------------------------------------------------------------
# Figure 6: data-scaling ablation (4 -> 7 training scenes, counter-7k held out).
# ---------------------------------------------------------------------------
def fig_data_scaling(out_dir):
    n_train = [4, 5, 6, 7]
    psnr = [23.23, 23.79, 24.26, 24.51]
    deltas = [None, 0.56, 0.47, 0.25]  # per-scene increment, annotated on points

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(n_train, psnr, '-o', color=C_KPCN, linewidth=2, markersize=7)
    for x, y, d in zip(n_train, psnr, deltas):
        ax.annotate(f'{y:.2f}', (x, y), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9)
        if d is not None:
            ax.annotate(f'+{d:.2f}', (x - 0.5, (y + psnr[n_train.index(x) - 1]) / 2),
                        ha='center', va='center', fontsize=8, color=C_ACCENT)
    ax.set_xlabel('# training scenes')
    ax.set_ylabel('held-out test PSNR (dB)')
    ax.set_title('Diminishing returns from data\n(motivates architecture, not more capture)',
                 fontsize=10)
    ax.set_xticks(n_train)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, 'fig_data_scaling.png')


# ---------------------------------------------------------------------------
# Figure 7: noise-level sweep (KPCN, counter-7k). Input vs denoised at 1/2/4 spp.
# ---------------------------------------------------------------------------
def fig_noise_levels(out_dir):
    spp = [1, 2, 4]
    noisy = [16.61, 19.54, 21.12]
    denoised = [27.17, 28.31, 28.89]

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(spp, noisy, '--s', color=C_NOISY, linewidth=2, markersize=7,
            label='noisy input')
    ax.plot(spp, denoised, '-o', color=C_KPCN, linewidth=2, markersize=7,
            label='KPCN denoised')
    for x, yn, yd in zip(spp, noisy, denoised):
        ax.annotate(f'+{yd - yn:.1f} dB', (x, (yn + yd) / 2),
                    textcoords='offset points', xytext=(8, 0),
                    fontsize=8, color=C_ACCENT, va='center')
    ax.set_xlabel('input samples per pixel (spp)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Denoiser does the most work at the noisiest input',
                 fontsize=10)
    ax.set_xticks(spp)
    ax.legend(fontsize=9, loc='center right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, out_dir, 'fig_noise_levels.png')


# ---------------------------------------------------------------------------
# Figure 8: quality vs latency (counter-7k, tuned baselines). Log-x.
# Note: KPCN latency is GPU; classical filters are CPU/numpy -> annotate.
# ---------------------------------------------------------------------------
def fig_quality_latency(out_dir):
    # (label, ms/frame, PSNR, color)
    pts = [
        ('Gaussian',      6.0,  25.83, C_CLASS),
        ('cross-bilat',   654., 26.36, C_CLASS),
        ('bilateral',     570., 26.47, C_CLASS),
        ('KPCN (ours)',   18.6, 27.17, C_KPCN),
    ]
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for label, ms, p, c in pts:
        ax.scatter(ms, p, s=90, color=c, edgecolor='black',
                   linewidth=0.6, zorder=3)
        dx = -10 if label == 'KPCN (ours)' else 8
        ha = 'right' if label == 'KPCN (ours)' else 'left'
        ax.annotate(label, (ms, p), textcoords='offset points',
                    xytext=(dx, 6), ha=ha, fontsize=9,
                    fontweight='bold' if 'ours' in label else 'normal')
    ax.set_xscale('log')
    ax.set_xlim(3, 1500)               # set BEFORE the band (log axis can't span 0)
    ax.axvspan(3, 33, color='green', alpha=0.07)  # <33 ms ~ real-time (30 fps)
    ax.axvline(33, color='green', alpha=0.4, linewidth=0.8, linestyle=':')
    ax.text(31, 25.85, '30 fps', fontsize=7, color='green', va='bottom', ha='right')
    ax.set_xlabel('latency per frame (ms, log scale)')
    ax.set_ylabel('held-out PSNR (dB)')
    ax.set_title('Best quality AND real-time (counter-7k)\nKPCN: GPU; classical: CPU/numpy',
                 fontsize=10)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    _save(fig, out_dir, 'fig_quality_latency.png')


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/figures')
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    fig_head_swap(args.out)
    fig_data_scaling(args.out)
    fig_noise_levels(args.out)
    fig_quality_latency(args.out)
    print('done.')


if __name__ == '__main__':
    main()
