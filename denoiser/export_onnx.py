"""Export a trained denoiser checkpoint to ONNX for the live in-browser demo.

The browser demo runs this graph with onnxruntime-web on the WebGPU backend, so
the whole point of this script is to find out -- on your machine, before any JS
is written -- whether the KPCN head's `F.unfold` + `softmax` survive the ONNX
round-trip and still match the PyTorch output numerically.

    python denoiser/export_onnx.py --ckpt results/denoiser/best.pt \
        --out web_demo/denoiser.onnx

By default H and W are dynamic axes, so the same .onnx runs at 256/384/512 in the
demo (the net is fully convolutional). Pass --size N to bake a fixed square size
instead (sometimes friendlier to web runtimes if dynamic shapes misbehave).

Verification: if onnxruntime is importable, we run the exported graph on CPU and
print the max abs difference vs. PyTorch. Anything <~1e-4 means the export is
faithful; a hard failure here (unsupported op) is exactly the KPCN risk we wanted
to surface early.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from model import UNetDenoiser


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='results/denoiser/best.pt')
    ap.add_argument('--out', default='web_demo/denoiser.onnx')
    ap.add_argument('--size', type=int, default=None,
                    help='bake a fixed square input size (e.g. 512). Default: '
                         'dynamic H/W so one graph serves any resolution.')
    ap.add_argument('--opset', type=int, default=17)
    args = ap.parse_args()

    # Export on CPU in fp32 -- deterministic and matches how ORT-web will run it.
    ckpt = torch.load(args.ckpt, map_location='cpu')
    ck_args = ckpt.get('args', {})
    base = ck_args.get('base', 32)
    head = ck_args.get('head', 'residual')
    ksize = ck_args.get('kernel_size', 11)
    model = UNetDenoiser(in_ch=4, out_ch=3, base=base, head=head, kernel_size=ksize)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'loaded {args.ckpt}: base={base}, head={head}, kernel_size={ksize}')

    # A size that is a multiple of 16 keeps the 4 pool/upconv stages exact.
    probe = args.size or 256
    if probe % 16 != 0:
        raise SystemExit(f'--size must be a multiple of 16 (4 downsamples), got {probe}')
    dummy = torch.rand(1, 4, probe, probe)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic batch is harmless; dynamic H/W is what lets the demo switch res.
    dynamic_axes = {'input': {0: 'N'}, 'output': {0: 'N'}}
    if args.size is None:
        dynamic_axes['input'].update({2: 'H', 3: 'W'})
        dynamic_axes['output'].update({2: 'H', 3: 'W'})

    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=['input'], output_names=['output'],
        opset_version=args.opset, dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    print(f'exported -> {out_path}  (opset {args.opset}, '
          f'{"dynamic H/W" if args.size is None else f"fixed {probe}x{probe}"})')

    # The new exporter writes weights to a sibling .onnx.data file. ORT-Web loads
    # from a single URL and will NOT auto-fetch that sidecar, so consolidate the
    # weights back inline into one self-contained file (and drop the sidecar).
    import onnx
    consolidated = onnx.load(str(out_path))  # pulls in external data from the dir
    onnx.save_model(consolidated, str(out_path), save_as_external_data=False)
    sidecar = out_path.with_suffix(out_path.suffix + '.data')
    if sidecar.exists():
        sidecar.unlink()
    print(f'consolidated weights inline -> single file {out_path} '
          f'({out_path.stat().st_size / 1e6:.1f} MB)')

    # ---- numerical verification against PyTorch (best-effort) ----
    with torch.no_grad():
        ref = model(dummy).numpy()
    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed -- skipping verification. '
              'Install with `pip install onnxruntime` to check op support/accuracy.')
        return
    sess = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    got = sess.run(['output'], {'input': dummy.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    print(f'verify (CPU EP): max|onnx - torch| = {diff:.2e}', end='  ')
    print('OK' if diff < 1e-3 else 'WARNING: large drift, inspect the graph')
    print(f'ops in graph: input {got.shape}, output range '
          f'[{got.min():.3f}, {got.max():.3f}]')


if __name__ == '__main__':
    main()
