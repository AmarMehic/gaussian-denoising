# Building on Arnes HPC

This is the renderer-side bring-up. The Python denoiser does not need HPC for
small-scale development.

## Prereqs (one time, on the login node)

```bash
ssh am5348@hpc-login.arnes.si
cd /d/hpc/home/$USER
git clone <this-repo-url> gaussian-denoising
cd gaussian-denoising
```

The StochasticSplats renderer is intentionally **not** vendored in this repo
(`.gitignore: renderer/`). It will be cloned into `renderer/stochasticsplats/`
by the build script.

The scene `.ply` files are also not in git. Either copy them up from your
laptop via `rsync`, or re-run the download commands from `plan.md`. The smoke
test expects at least `data/scenes/bonsai/point_cloud/iteration_30000/point_cloud.ply`.

## Build

vcpkg compiles ~15 third-party libs from source on the first run (20-40 min).
Do this on a compute node, not the login node:

```bash
# Grab an interactive CPU node
srun --partition=all --time=2:00:00 --cpus-per-task=8 --mem=16G --pty bash

# Inside the node:
cd /d/hpc/home/$USER/gaussian-denoising
source scripts/hpc_env.sh
bash scripts/hpc_build.sh
```

On success the binary lives at `renderer/stochasticsplats/build/splatapult`.

## Headless smoke test

Now ask SLURM for a node with a GPU and verify the binary actually runs
without a display:

```bash
srun --partition=gpu --gres=gpu:1 --time=00:15:00 --cpus-per-task=4 --mem=16G --pty bash

cd /d/hpc/home/$USER/gaussian-denoising
source scripts/hpc_env.sh
bash scripts/hpc_smoke.sh
```

The script tries `SDL_VIDEODRIVER=offscreen`, then falls back. Report back
which (if any) driver works — that determines whether the next chunk is a
straight shader change or also includes an EGL port.

## What to send back

- Full output of `bash scripts/hpc_build.sh 2>&1 | tee build.log`
- Full output of `bash scripts/hpc_smoke.sh 2>&1 | tee smoke.log`
- `nvidia-smi` output from the GPU node so we know which GPU you got

## Common failure modes (anticipated)

| Symptom | Likely cause | Fix |
|---|---|---|
| `find_package(X11) Could NOT find X11` | X11 dev headers missing on compute node | Try a different compute node (`-w wn212`), or install xorg via vcpkg overlay |
| `bootstrap-vcpkg.sh: line N: cmake: command not found` | env not sourced in subshell | `source scripts/hpc_env.sh` first |
| vcpkg build of `openxr-loader` fails | known with older GCC | We can disable OpenXR — it's only used for VR, not needed for batch render |
| `splatapult` exits with "Could not create window" under `offscreen` | SDL2 from vcpkg built without offscreen backend | Set `VCPKG_FEATURES` to include offscreen, or use EGL port |
| `splatapult` segfaults immediately under `offscreen` | OpenGL context creation needs an actual display | Move to EGL port (chunk 2b) |
