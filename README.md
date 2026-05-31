# Real-Time Denoising of Stochastic Gaussian-Splatting Renders

Seminar project for **Advanced Computer Graphics** (UL FRI). A kernel-predicting
U-Net (KPCN head) denoises 1-spp stochastic-transparency 3D Gaussian-splatting
renders in real time, using the noisy RGB plus a depth auxiliary channel.

## Layout

| Path | What |
|------|------|
| `denoiser/` | PyTorch denoiser: model, training, evaluation, classical baselines |
| `webgpu-splatting-dithering-nrg/` | WebGPU stochastic-transparency splatting renderer + capture tool (by Žiga Lesar) |
| `scripts/` | Env setup and Slurm jobs for the Arnes HPC |

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train (leave-one-scene-out; held-out scene = test set)
python denoiser/train.py --data data/renders --head kpcn --holdout garden

# evaluate PSNR/SSIM + per-frame latency
python denoiser/evaluate.py --ckpt results/denoiser/best.pt --data data/renders

# write a noisy | denoised | clean comparison strip
python denoiser/inference.py --ckpt results/denoiser/best.pt
```

`evaluate.py` supports `--scale` (denoise at reduced resolution and upsample) and
`--half` (fp16 conv body on CUDA/MPS) for a latency/quality knob.

## Live browser demo

A live demo renders a scene at 1 spp and denoises it in real time in the browser
with onnxruntime-web (WebGPU backend) — the same KPCN model, exported to ONNX:

```bash
# export the trained checkpoint to a self-contained ONNX (web_demo/denoiser.onnx)
python denoiser/export_onnx.py --ckpt results/denoiser/best.pt

# serve the repo root, then open the demo in Chrome
python3 scripts/capture_server.py
# http://localhost:8000/webgpu-splatting-dithering-nrg/demo.html?scene=garden-7k
```

Use `?size=256|384|512` and `?spp=N` to trade quality for speed. The prebuilt
ONNX is also attached to the GitHub Release.

## Data and weights

The renders (`data/`) and trained checkpoints are not committed. The trained KPCN
weights (PyTorch `.pt` and the browser `.onnx`) are published as a
[GitHub Release](../../releases/tag/checkpoints-v1).

## License

MIT (see [LICENSE](LICENSE)). The renderer in `webgpu-splatting-dithering-nrg/`
(excluding the `capture.*`/`demo.*` files added here) is by Žiga Lesar and its
rights remain with its author.

## Credits

The renderer in `webgpu-splatting-dithering-nrg/` was written by teaching
assistant **Žiga Lesar** and is built on the UL FRI WebGPU engine framework.
Scenes are from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset.
