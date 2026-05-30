# Project notes — Denoising stochastic Gaussian-splatting renders

Working notes for the seminar report. Facts, design decisions, and numbers
collected as we build the pipeline. Final headline results get filled in once
the full HPC training run completes (see **Results** → marked TODO).

---

## 1. Problem & pipeline

Goal: take the noisy output of a **stochastic-transparency WebGPU Gaussian
splatting renderer** (`webgpu-splatting-dithering-nrg`) and denoise it with a
learned U-Net, so a cheap low-sample render looks like an expensive converged
one.

Pipeline: **capture dataset (browser/WebGPU)** → **train U-Net (PyTorch, Arnes
HPC/CUDA)** → **evaluate (PSNR/SSIM + latency vs. a classical baseline)** →
report.

Why the noise exists: the renderer uses **stochastic transparency / dithering** —
each frame randomly keeps or discards Gaussians according to opacity. A single
frame (1 spp) is a noisy random draw; the clean image is the average of many
draws.

### Nature of the noise — spatial Monte Carlo, *not* motion-only
A common misconception: "this noise only appears when objects move." That is a
half-truth that conflates two different things, and it does **not** describe our
problem.

- **What we denoise is spatial Monte Carlo noise.** Stochastic transparency
  replaces sorted alpha-blending with a random keep/discard decision per
  fragment, so each pixel is a Monte Carlo *estimate* of the true color. At 1 spp
  the variance is high → heavy per-pixel noise, **present in every single frame,
  static or moving.** Our dataset is literally *static* poses (1/2/4 spp noisy vs
  400-spp clean) — nothing moves, and the noise is fully there.
- **Where "only when moving" comes from:** real-time stochastic renderers often
  hide the noise with *temporal accumulation* (TAA-style averaging across
  frames). On a static scene that averaging is free and converges to clean, so it
  *looks* noise-free. Under motion, reprojection/accumulation breaks
  (disocclusion, ghosting), the average resets, and the per-frame noise becomes
  visible again. So a viewer perceives "noise appears when moving," but the noise
  was always there per frame — motion just removes the temporal crutch hiding it.
- **Why this matters for us (report point):** a per-frame learned denoiser is the
  *alternative* to temporal accumulation. It cleans a single noisy frame
  directly, so it behaves identically whether the scene is static or moving — no
  motion vectors, no reprojection, no ghosting under motion. It degrades
  gracefully under motion *because* it doesn't depend on temporal accumulation.

---

## 2. Dataset

- **8 scenes** (MipNeRF360, 7k-splat versions): `bonsai`, `bicycle`, `stump`,
  `counter`, `garden`, `kitchen`, `playroom`, `room` (mix of indoor + outdoor).
- **80 camera poses / scene → 640 frames total.** Diversity across scenes was
  prioritised over raw frame count (more scenes generalise better than more
  views of the same scene).
- **Resolution:** 512×512.
- **Per pose, 6 files:**
  - `_noisy1.png`, `_noisy2.png`, `_noisy4.png` — noisy inputs at 1 / 2 / 4 spp
    (accumulator snapshot at that sample count).
  - `_clean.png` — target, 400-spp running-average accumulation.
  - `_depth.f32` — raw linear depth, row-major float32, 0 = background
    (512×512×4 = 1,048,576 bytes).
  - `_depth.png` — 8-bit depth preview (not used for training).
- **Size on disk:** ~2.5 GB (transferred to HPC as ~2.1 GB, PNGs compress).
- **Depth is rendered deterministically** in a separate pass (noise-free).

### Capture harness details (for methods section)
- Camera orbits a per-scene pivot found by `scripts/find_pivot.py`: robust
  trimmed centroid (4 passes, keep inner 60%) for scene centering, then
  densest-voxel + radius-ball centroid for the subject pivot.
- Convention: **+Y is down** (COLMAP / MipNeRF360); up-vector = [0, −1, 0].
- `.splat` format: 32 bytes/splat — position f32×3, scale f32×3, rgba u8×4,
  rotation u8×4.
- Engineering fix worth a sentence: Chrome pauses `requestAnimationFrame` in
  backgrounded tabs, stalling headless capture. Replaced the rAF yield with a
  **MessageChannel postMessage** yield, which is delivered (and not throttled)
  in background tabs.

---

## 3. Network input design — why 4 channels (RGB + depth), not 7

Input = **noisy RGB (3) + per-image normalized depth (1) = 4 channels.**
Output = clean RGB (3).

### Depth channel
- Per-image robust normalization to [0,1]: divide by the 99th-percentile depth,
  clip, background stays 0. Scenes differ wildly in absolute depth
  (bonsai ~4–28, stump much deeper), so a per-image scale keeps the channel in a
  consistent range while preserving relative structure.
- Depth is a **safe guide**: it is noise-free *and* it encodes geometry (where
  surfaces are), not color — so it helps without leaking the answer.

### Why albedo is **excluded** (label leakage) — key design decision
Albedo = the intrinsic surface/material color, before lighting. In ordinary
path-tracing denoisers (OIDN, OptiX, Mara 2017) albedo is a prized guide,
because there RGB is noisy (the *lighting* integral is undersampled) while
albedo is noise-free (a deterministic primary-hit lookup) — two genuinely
different signals.

In **our splatting renderer there is no separate lighting term** — a Gaussian's
stored color *is* its appearance. So:
- A **clean albedo** ≈ the converged clean target → feeding it in lets the model
  cheat by near-copying albedo → output. Meaningless PSNR. **(leakage)**
- A **1-spp (noisy) albedo** doesn't leak, but it's produced by the same
  dithering of the same colors → it's essentially a duplicate of the noisy RGB
  input, carrying no new *clean* structure. **(useless / redundant)**

> One-liner: albedo only helps when it's *clean*, and in this renderer a clean
> albedo ≈ the *answer*; make it noisy to avoid leaking and it's just a copy of
> the input. Depth is the only guide that is both informative (clean) and
> non-leaking (geometry, not color).

Confirmed visually: a collaborator's separate toy renderer shows the "Albedo"
panel nearly identical to the 64-spp "Reference" panel — a direct picture of the
leakage. (Caveat: in a renderer with a real lighting integral, albedo ≠
reference and would be a legitimate, valuable guide.)

---

## 4. Model

- `UNetDenoiser` (denoiser/model.py): 4→3 channels, **base width 32, ~7.76M
  parameters**.
- Standard encoder/decoder U-Net: 4 downsampling levels + bottleneck, skip
  connections, `DoubleConv` blocks with **GroupNorm(8)** + ReLU.
- **Residual parameterization:** predicts a residual added to the noisy RGB
  input (the identity / no-op is easy to represent; the net only learns the
  noise to remove).
- Constraint: input H,W must be divisible by 16 (128 crop in training, full 512
  at eval — both OK).

---

## 5. Training setup

- Optimizer **Adam**, lr **1e-4**, **L1** loss, **cosine-annealing** LR schedule
  (T_max = epochs), **AMP** mixed precision on CUDA.
- batch **16**, crop **128**, **100 epochs** (default).
- **Noise-level augmentation:** each training sample draws a *random* level from
  {1, 2, 4} spp per access → robustness across noise strengths. Validation/test
  always use the **worst case, 1 spp** (`min` level), so all reported metrics are
  on the hardest input.
- Spatial augmentation: random crop + horizontal/vertical flips + 90° rotations.
- Best-val-PSNR checkpoint saved (`best.pt`); CSV log of per-epoch
  loss/PSNR/SSIM/lr/time (`train_log.csv`).

### Evaluation protocol — leave-one-scene-out
- **Held-out scene = `counter-7k`** is the *entire* test set (unseen
  geometry/content → a true generalization number). The other 7 scenes are split
  train/val.
- Split sizes: **504 train / 56 val / 80 test.**
- Rationale: a per-scene pose split would only test novel *views* of scenes the
  model trained on (weaker claim, easy to over-read). Leave-one-scene-out is the
  defensible generalization metric. Holdout scene is a one-line change.
- The classical baseline is evaluated on the **same** held-out scene for an
  apples-to-apples comparison.

---

## 6. Classical baseline (training-free reference)

`denoiser/baselines.py` — pure numpy + PIL (runs without a GPU). Methods,
weakest → strongest:
1. `noisy` — the 1-spp input itself (the floor every method must beat).
2. `gauss` — fixed Gaussian blur (edge-blind).
3. `bilateral` — color-only bilateral (edge-aware on RGB).
4. `xbilat` — **depth-guided cross-bilateral filter (Mara 2017 aligned)**: same
   spatial + range weighting, but the range term also includes the depth
   channel, so it stops smoothing across depth discontinuities. Uses the same
   noisy RGB + depth inputs as the U-Net.

**Why this set (rationale for the report).** The baselines aren't arbitrary —
they form a **ladder where each rung adds exactly one capability**, so the
comparison reveals *what kind* of work the network is doing, not just whether it
wins:

| baseline   | capability it adds        | question it answers                          |
|------------|---------------------------|----------------------------------------------|
| `noisy`    | nothing (passthrough)     | input SNR — the floor improvement is measured against |
| `gauss`    | pure low-pass smoothing   | does this problem just need *blurring*?      |
| `bilateral`| edge-awareness (RGB range)| is *edge-preserving* smoothing enough?       |
| `xbilat`   | geometry-awareness (depth)| can a hand-built filter using the *same depth input* do it? |

Reading the ladder: *blur → edge-aware → geometry-aware → learned*, and you can
point to exactly where the net lands and which classical capability it surpasses.

- `gauss` is the "do you even need a net?" check. It exposed the **residual** head
  as basically a fancy blur (residual *lost* to gauss on garden); KPCN beating
  gauss means KPCN is doing something a blur can't.
- `xbilat` is the most important rung: it's the **fair classical analog of the
  U-Net**, using the *same* noisy-RGB + depth inputs. Beating it means "given
  identical inputs, the *learned* per-pixel kernel exploits geometry better than
  the best hand-tuned filter" — a far stronger claim than beating an RGB-only filter.

This set is also the **canonical Monte-Carlo-rendering lineage**: guided/cross-
bilateral over a G-buffer (depth/normal) is *the* real-time classical reference
(Mara 2017), and is exactly what the kernel-predicting papers (Bako 2017 / KPCN)
benchmarked against — so it's the defensible comparison for this problem.

**Deliberately excluded:** temporal accumulation / TAA (needs multiple frames —
out of scope for single-frame denoising); OIDN / neural denoisers (a *different*
axis — "our net vs another net," not "net vs training-free classical"); median
filters (MC noise is variance-like, not impulse, so the bilateral family fits the
noise model better).

**Key implementation finding:** a naive bilateral *fails* at 1 spp because the
noise variance is so high the color-range term mistakes noise for edges and
preserves it (≈19 dB, barely above noisy). Fix: compute the range/guide weights
from a **pre-smoothed guide image** (joint-bilateral trick) → ≈27 dB. Worth
mentioning as a non-trivial detail.

**Preliminary baseline numbers** (bonsai, 6 frames, quick check — *not* the
final held-out-scene numbers):

| method     | PSNR (dB) | SSIM   | sec/img |
|------------|-----------|--------|---------|
| noisy      | 18.56     | 0.3040 | —       |
| gauss      | 27.54     | 0.7284 | 0.007   |
| bilateral  | 27.13     | 0.7181 | 0.69    |
| xbilat     | 27.09     | 0.7234 | 0.81    |

Note: depth-guided `xbilat` gives the best SSIM even on bonsai (which is
depth-flat); the gap should widen on depth-discontinuous scenes (garden,
bicycle). *TODO: replace with held-out counter-7k numbers from the full run.*

---

## 7. HPC environment (Arnes)

- Cluster has EasyBuild modules. Using
  **`PyTorch/2.1.2-foss-2023a-CUDA-12.1.1`** (matches `torch>=2.1`, ships CUDA
  12.1, Python 3.11, numpy).
- Bare cluster Python has **no torch** — must load the module.
- Pillow is layered into a `--system-site-packages` venv
  (`$HOME/denoise-venv`); torch + numpy inherited from the module (no multi-GB
  reinstall). One-time setup: `scripts/setup_env.sh` (run on login node).
- Verified stack: **torch 2.1.2 | numpy 1.25.1 | Pillow 10.0.0 | CUDA 12.1**.
- SLURM: `gpu` partition, 1 GPU, 8 CPUs, 32 GB, walltime 3h (training only
  needs ~10 min; short walltime backfills onto a node faster).
- Data is gitignored; synced to HPC via `rsync` (not git).

---

## 8. Results

### Smoke test (2 epochs — pipeline validation only, NOT quality)
- `cuda True`, model 7.76M params, split 504/56/80.
- **~5 s / epoch** on the Arnes GPU → full 100-epoch run ≈ **8–12 min**.
- **Inference latency: 15.4 ms / 512×512 frame (warm median) → ~65 fps.**
- counter-7k after 2 epochs: noisy 16.61 dB / 0.3001 → denoised 17.03 dB /
  0.2888. Meaningless at 2 epochs (val PSNR still climbing, SSIM below noisy
  because half-trained output is mushy) — confirms only that the pipeline runs.

### Full run (100 epochs) — held-out generalization, 2 scenes

Headline finding: **the residual U-Net underperforms the classical filters on
held-out scenes** — reproducibly, across two holdout scenes and two loss
variants. The bottleneck is architecture/data, not the loss (the loss tweak did
not close the gap). Latency 15.4 ms / 512×512 frame (~65 fps) on the GPU.

**counter-7k held out** (noisy input 16.61 dB / 0.300):
| method                    | PSNR (dB) | SSIM   | latency |
|---------------------------|-----------|--------|---------|
| U-Net residual (pure L1)  | 24.64     | 0.615  | 15.4 ms |
| U-Net residual (L1+0.2SSIM)| 24.32    | 0.610  | 15.4 ms |
| gauss (tuned)             | 25.83     | 0.6635 | —       |
| bilateral (tuned)         | 26.47     | 0.6852 | 570 ms  |
| xbilat (tuned)            | 26.36     | 0.6816 | 654 ms  |
| **U-Net KPCN (pure L1)**  | **27.17** | **0.724** | 18.6 ms |

Baselines are **tuned on the train split** (grid-search, then frozen and applied
to the held-out test scene — never tuned on test). Frozen params: gauss σ=1.5;
bilateral σ_s=4, σ_r=0.1, guide σ=1.5; xbilat σ_s=4, σ_r=0.1, σ_d=0.2, guide σ=1.5.
Tuning lifted the best classical from 26.00→26.47 dB, so this is the strongest
fair version of the competition.

**HEADLINE: the kernel-prediction (KPCN) head is the result.** Swapping ONLY the
output head (residual→KPCN), same data/loss, lifted the same U-Net by
**+2.53 dB / +0.109 SSIM** — more than the loss tweak or the whole 4→7 data curve
(+1.28 dB) combined. KPCN at 27.17 dB **beats the best train-tuned classical filter
(bilateral 26.47) by +0.70 dB / +0.039 SSIM**, while being **~30× faster**
(18.6 ms GPU vs 570 ms numpy) and real-time (~54 fps). The residual U-Net *lost*
to a Gaussian blur; the KPCN head *beats the depth-guided cross-bilateral*. SSIM
gained the most (0.615→0.724), confirming the mechanism: a normalized per-pixel
kernel can only redistribute real input pixels, so it is structurally
edge-preserving and cannot produce the L1 smear. This is the architecture lever
the diminishing-returns ablation pointed to.

**garden-7k held out** (noisy input 16.24 dB / 0.260):
| method                | PSNR (dB) | SSIM   | latency |
|-----------------------|-----------|--------|---------|
| U-Net residual (L1+0.2SSIM)| 24.01 | 0.568  | 15.4 ms |
| gauss (tuned)         | 24.80     | 0.5620 | —       |
| bilateral (tuned)     | 24.92     | 0.5490 | 592 ms  |
| xbilat (tuned)        | 24.82     | 0.5497 | 679 ms  |
| **U-Net KPCN (pure L1)** | **25.23** | **0.611** | 18.6 ms |

Baselines **tuned on the train split** (same protocol as counter): tuning fixed
the previously mistuned bilateral (24.37→24.92 dB, 0.487→0.549 SSIM). After
tuning the best classical is split — **bilateral leads PSNR (24.92), gauss leads
SSIM (0.562)** — and KPCN still beats *both*: +0.31 dB over best-PSNR bilateral,
+0.049 SSIM over best-SSIM gauss.

**Generalization confirmed.** KPCN wins on garden too: the residual head *lost*
to a plain Gaussian here (24.01 < 24.80), but the KPCN head *beats* every
train-tuned classical on both metrics. Margin is smaller than counter (garden is
more textured/harder), but it's a clean two-metric win on a second, very
different (outdoor) unseen scene → the result is robust, not scene-specific.

**Cross-scene summary (KPCN vs best classical):**
| held-out | KPCN PSNR/SSIM | best classical | margin |
|----------|----------------|----------------|--------|
| counter (indoor)  | 27.17 / 0.724 | bilateral 26.47 / 0.685 (tuned) | +0.70 dB / +0.039 |
| garden (outdoor)  | 25.23 / 0.611 | bilateral 24.92 / gauss 0.562 (tuned) | +0.31 dB / +0.049 |

Both rows use **train-tuned** classical baselines (grid-searched on the train
split, frozen, applied to the held-out test scene). KPCN wins both metrics on
both scenes against the strongest fair version of the competition.

**Noise-level sweep (KPCN, counter-7k held out, `evaluate.py --level`):**
| input spp | noisy PSNR/SSIM | denoised PSNR/SSIM | improvement |
|-----------|-----------------|--------------------|-------------|
| 1 (worst) | 16.61 / 0.300   | 27.17 / 0.724      | +10.56 dB / +0.424 |
| 2         | 19.54 / 0.403   | 28.31 / 0.756      | +8.78 dB / +0.353  |
| 4         | 21.12 / 0.468   | 28.89 / 0.771      | +7.77 dB / +0.303  |

Reading: denoised quality rises with samples (27.2→28.9 dB), but the *gain over
the input shrinks* (+10.6→+7.8 dB) — the denoiser does the most work at the
noisiest 1-spp setting, exactly the cheap-render regime we target. Latency is
flat (~16 ms) across levels — denoising cost is independent of input noise. The
model trained with random {1,2,4}-spp augmentation, so all three are in-distribution.

Notes for the report:
- **SSIM loss is a negative result.** L1 + 0.2·(1−SSIM) was *worse* than pure L1
  on counter (PSNR & SSIM & val all down). Reverted to pure L1. (Caveat earlier:
  a first SSIM-loss run was broken by fp16 catastrophic cancellation in the
  variance terms under AMP autocast → inf/nan grads silently skipped by
  GradScaler → model under-trained to 21.6 dB. Fixed by computing the SSIM term
  in fp32 outside autocast; the 24.32 above is the *correct* SSIM-loss number.)
- **[RESOLVED] Baseline tuning done.** Earlier the garden bilaterals were
  mistuned (0.49 SSIM, worse than gauss). The `--tune` mode now grid-searches each
  filter on the *train* split, freezes the params, and applies them to the held-out
  test scene — fixing garden bilateral to 24.92/0.549. KPCN beats every train-tuned
  classical on both metrics on both scenes (see tables above). This refers to the
  superseded *residual* U-Net; the headline result is the KPCN head.
- Why the U-Net trails: residual/direct RGB regression with L1 is blur-prone;
  with only 7 training scenes the learned prior is data-starved (a hand-coded
  edge/depth filter needs no data). Two levers under investigation:
  (1) **kernel-prediction head** (KPCN-style: predict a normalized per-pixel
  kernel applied to noisy RGB — cannot hallucinate blur, only redistribute
  pixels → edge-preserving); (2) **more training scenes** (see ablation below).

### Data-scaling ablation (4 → 7 training scenes, counter-7k held out)
`scripts/ablation_scenes.sh`, ssim_weight=0.2, 100 epochs each.

| n_train | test PSNR | Δ      | test SSIM |
|---------|-----------|--------|-----------|
| 4       | 23.23     | —      | 0.563     |
| 5       | 23.79     | +0.56  | 0.585     |
| 6       | 24.26     | +0.47  | 0.608     |
| 7       | 24.51     | +0.25  | 0.618     |

**Interpretation: data-limited but with strongly diminishing returns.** Test PSNR
is still climbing at 7 scenes (not saturated), but the per-scene increment is
roughly halving (+0.56 → +0.47 → +0.25). Extrapolated, each extra scene buys
~0.1–0.2 dB and falling — so reaching the classical bilateral baseline (26.0)
from data alone would need an unrealistic number of new scenes. The
generalization gap also persists/widens (N=7: val 26.07 vs test 24.51 = 1.56 dB),
implicating the model's inductive bias, not just data quantity.

**Conclusion → change the architecture, not the dataset.** This motivates the
kernel-prediction (KPCN) head over endless capture. Good report figure: the 4→7
curve + "diminishing returns justify an architectural change." (Note: this curve
ran at ssim_weight=0.2; pure L1 shifts it up ~0.4 dB — the N=7 pure-L1 run was
24.94 vs 24.51 here. Shape is unchanged.)

---

## 9. Talking points / caveats for the report

- Diversity > count: 8 scenes × 80 poses, not 1000 frames of fewer scenes.
- Multi-noise training (1/2/4 spp) → one model robust across sample counts;
  evaluated worst-case at 1 spp.
- Leave-one-scene-out = honest generalization, not memorized scenes.
- Albedo exclusion is a *consequence of the splat setting* (no lighting
  integral), not a general rule — flag this so it doesn't look like an oversight.
- Real-time-capable: ~15 ms/frame inference.
