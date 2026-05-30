"""Torch-free data helpers shared by the dataset and the classical baselines.

These functions only depend on numpy + Pillow so that the baseline script
(denoiser/baselines.py) can run on a machine without PyTorch installed. The
PyTorch Dataset in dataset.py imports everything it needs from here.
"""

import random
import re
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 512  # square render resolution from the capture harness

_NOISY_RE = re.compile(r'_noisy(\d+)\.png$')


def _load_rgb(path):
    img = Image.open(path).convert('RGB')
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HxWx3 in [0,1]
    return arr


def _load_noisy(base, level):
    """Load the noisy frame at a given spp level, with legacy fallback."""
    p = Path(f'{base}_noisy{level}.png')
    if not p.exists():
        p = Path(f'{base}_noisy.png')  # legacy single-level naming
    return _load_rgb(p)


def _load_depth(path, size=SIZE):
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size != size * size:
        side = int(round(raw.size ** 0.5))
        raw = raw[: side * side]
        size = side
    return raw.reshape(size, size)  # HxW, 0 = background


def _normalize_depth(depth):
    """Per-image robust normalization to [0,1]; background stays 0.

    Scenes differ wildly in absolute depth (bonsai ~4-28, stump much deeper),
    so a per-image scale keeps the depth channel in a consistent range while
    preserving the relative structure the model actually uses.
    """
    valid = depth[depth > 0]
    if valid.size == 0:
        return np.zeros_like(depth)
    scale = np.percentile(valid, 99.0)
    if scale <= 0:
        scale = valid.max()
    out = np.clip(depth / scale, 0.0, 1.0)
    out[depth <= 0] = 0.0
    return out.astype(np.float32)


def _noisy_levels(scene_dir, stem):
    """Available noise levels for a sample, e.g. {1,2,4}. Falls back to the
    legacy single '<stem>_noisy.png' (treated as level 1) if present."""
    levels = []
    for p in scene_dir.glob(f'{stem}_noisy*.png'):
        m = _NOISY_RE.search(p.name)
        if m:
            levels.append(int(m.group(1)))
    if not levels and (scene_dir / f'{stem}_noisy.png').exists():
        levels.append(1)  # legacy naming
    return sorted(levels)


def discover_samples(renders_root, scenes=None):
    """Return a sorted list of (scene, id, base_path, [noise_levels]) per sample."""
    root = Path(renders_root)
    samples = []
    scene_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if scenes:
        wanted = set(scenes)
        scene_dirs = [p for p in scene_dirs if p.name in wanted]
    for scene_dir in scene_dirs:
        for clean in sorted(scene_dir.glob('*_clean.png')):
            stem = clean.name[: -len('_clean.png')]
            base = scene_dir / stem
            levels = _noisy_levels(scene_dir, stem)
            if levels and (scene_dir / f'{stem}_depth.f32').exists():
                samples.append((scene_dir.name, stem, base, levels))
    return samples


def split_samples(samples, val_frac=0.1, test_frac=0.1, seed=0):
    """Deterministic per-scene split so every scene appears in each subset."""
    by_scene = {}
    for s in samples:
        by_scene.setdefault(s[0], []).append(s)

    train, val, test = [], [], []
    rng = random.Random(seed)
    for scene in sorted(by_scene):
        items = sorted(by_scene[scene], key=lambda x: x[1])
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(round(n * test_frac)))
        n_val = max(1, int(round(n * val_frac)))
        test += items[:n_test]
        val += items[n_test:n_test + n_val]
        train += items[n_test + n_val:]
    return train, val, test
