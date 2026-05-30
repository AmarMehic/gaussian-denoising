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

### Full run (100 epochs) — **TODO**
Fill in from `results/denoiser/{train_log.csv, slurm-*.out}`:
- Final held-out **counter-7k**: noisy baseline PSNR/SSIM, U-Net PSNR/SSIM,
  improvement.
- U-Net vs. cross-bilateral baseline on counter-7k (headline comparison).
- Inference latency.
- Training curve (loss / val PSNR vs. epoch).

---

## 9. Talking points / caveats for the report

- Diversity > count: 8 scenes × 80 poses, not 1000 frames of fewer scenes.
- Multi-noise training (1/2/4 spp) → one model robust across sample counts;
  evaluated worst-case at 1 spp.
- Leave-one-scene-out = honest generalization, not memorized scenes.
- Albedo exclusion is a *consequence of the splat setting* (no lighting
  integral), not a general rule — flag this so it doesn't look like an oversight.
- Real-time-capable: ~15 ms/frame inference.
