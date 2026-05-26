# Building on Arnes HPC (Apptainer-based)

The C++/OpenGL renderer is built and run inside an **Apptainer container**
(Ubuntu 22.04 + system deps via apt). This sidesteps the CentOS-7 / vcpkg
incompatibilities on the HPC nodes. The Python denoiser side runs natively
on HPC modules — no container needed there.

## One-time setup on HPC

```bash
ssh am5348@hpc-login.arnes.si
cd /d/hpc/home/$USER
git clone <repo-url> Developer/gaussian-denoising   # adjust path
cd Developer/gaussian-denoising
```

Scene `.ply` files are not in git — `rsync` them up from your Mac (see
`README.md` / chat history for the command).

## Build

The container build itself is heavy (~10-20 min, ~1 GB .sif). Do it on a
compute node, not login:

```bash
srun --partition=all --time=2:00:00 --cpus-per-task=8 --mem=16G --pty bash
cd /d/hpc/home/$USER/Developer/gaussian-denoising
bash scripts/hpc_build.sh
```

`hpc_build.sh` orchestrates three steps:
1. Clone `ubc-vision/stochasticsplats` if missing
2. Build `containers/renderer.sif` if missing (the long one-time step)
3. Build the renderer binary inside the container via `scripts/build_renderer.sh`

The binary lands at `renderer/stochasticsplats/build/splatapult`. It's an
ELF compiled against Ubuntu 22.04 libs, so it only runs via
`apptainer exec --nv containers/renderer.sif ...`.

## Headless smoke test

On a GPU node:

```bash
srun --partition=gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=4 --mem=16G --pty bash
cd /d/hpc/home/$USER/Developer/gaussian-denoising
bash scripts/hpc_smoke.sh
```

Tries `SDL_VIDEODRIVER=offscreen` first; reports the winning driver.

## What goes where

```
containers/
  renderer.def         # Apptainer recipe (Ubuntu 22.04 + deps)
  renderer.sif         # built artifact (gitignored, ~1 GB)

scripts/
  hpc_build.sh         # orchestrator: clone + container + renderer build
  build_container.sh   # builds renderer.sif from the .def
  build_renderer.sh    # builds splatapult INSIDE the container
  patch_renderer.sh    # applies our overlay (no-X11, no-OpenXR)
  hpc_smoke.sh         # GPU-side headless smoke test (uses container)
  hpc_env.sh           # native HPC modules — only needed for PyTorch side

  renderer_overlay/    # source files that shadow upstream renderer
    CMakeLists.txt     # no X11, no OpenXR
    src/...            # maincontext.h, sdl_main.cpp, xrbuddy.{h,cpp} stubs
```

## What to send back after a build

```bash
bash scripts/hpc_build.sh 2>&1 | tee build.log
tail -50 build.log
```

## If something breaks

| Symptom | Try |
|---|---|
| `apptainer: command not found` | `which apptainer` — should be `/usr/bin/apptainer`. If missing, `module avail apptainer` |
| `apptainer build` fails with "no space left" | `$TMPDIR` may be small; `export APPTAINER_TMPDIR=/d/hpc/home/$USER/.apptainer_tmp && mkdir -p $APPTAINER_TMPDIR` |
| `cmake` can't find SDL2/glm/Eigen3 inside container | Inspect: `apptainer exec containers/renderer.sif dpkg -l \| grep -E 'sdl2\|glm\|eigen'` |
| Renderer segfaults under `--nv` | Host GPU drivers mismatch container GL libs. We can pin Mesa/OpenGL versions or switch to `nvidia-egl` runtime |
| `cuda` operations not visible | `--nv` flag missing from the apptainer exec call |
