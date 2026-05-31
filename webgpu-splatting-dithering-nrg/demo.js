// Live KPCN denoiser demo.
//
// Renders a Gaussian-splatting scene at 1 spp (the noisy, stochastic-transparency
// output), feeds noisy RGB + a normalized depth channel through the trained KPCN
// U-Net via onnxruntime-web (WebGPU backend), and shows noisy | denoised live as
// you orbit. This is the real-time inference path from the report, running in the
// browser on the same machine as the renderer.
//
// Reuses the renderer modules and scene-centering from capture.js verbatim; the
// only new parts are (1) the per-frame ORT inference and (2) reproducing the
// dataset's depth normalization (`_normalize_depth`) in JS so the network sees
// the same input distribution it was trained on.
//
// Serve from the project root (scripts/capture_server.py) and open:
//   http://localhost:8000/webgpu-splatting-dithering-nrg/demo.html?scene=garden-7k
//
// onnxruntime-web is loaded from a CDN, so this page needs internet access the
// first time (the wasm/js artifacts then cache).

import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs';

import { mat4, vec3 } from 'glm';
import { Camera, Node, Transform } from 'engine/core.js';
import { parseSplats } from './parseSplats.js';
import { Splat } from './Splat.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';
import { DepthRenderer } from './DepthRenderer.js';

// ---- Parameters (override via URL) ----
const params = new URLSearchParams(location.search);
const SCENE = params.get('scene') ?? 'garden-7k';
const SIZE = Number(params.get('size') ?? 512);   // must match the exported ONNX size
const SPP = Math.max(1, Number(params.get('spp') ?? 1)); // stochastic passes per input frame
const MODEL_URL = params.get('model') ?? '/web_demo/denoiser.onnx';

// Per-scene framing copied from capture.js (orbit pivot + camera distance).
const SCENE_PRESETS = {
    'bonsai-7k': { tx: 2.7, ty: -2.4, tz: -2.11, dist: 5.0 },
    'bicycle-7k': { tx: 0.02, ty: 2.60, tz: 0.03, dist: 4.0 },
    'stump-7k': { tx: 0.38, ty: 4.17, tz: -1.63, dist: 4.4 },
    'counter-7k': { tx: -0.04, ty: -0.05, tz: 0.71, dist: 5.5 },
    'garden-7k': { tx: -1.61, ty: 0.43, tz: -1.11, dist: 4.6 },
    'kitchen-7k': { tx: 0.1, ty: 0.0, tz: 0.35, dist: 5.5 },
    'playroom-7k': { tx: 0, ty: 0, tz: 0, dist: 6.0 },
    'room-7k': { tx: -2.64, ty: 0.49, tz: -0.1, dist: 5.0 },
};
const preset = SCENE_PRESETS[SCENE] ?? {};

const logEl = document.getElementById('log');
const statsEl = document.getElementById('stats');
function log(msg) { logEl.textContent += '\n' + msg; console.log(msg); }

// ---- Device ----
if (!navigator.gpu) {
    log('WebGPU is not available in this browser. Use Chrome/Edge 113+ or Safari TP.');
    throw new Error('no WebGPU');
}
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredFeatures: ['float32-blendable'] });

// ---- Load + center scene (verbatim from capture.js) ----
log(`Loading scene "${SCENE}" ...`);
const arrayBuffer = await fetch(`../data/scenes/${SCENE}.splat`).then(r => {
    if (!r.ok) throw new Error(`Failed to fetch ${SCENE}.splat (${r.status})`);
    return r.arrayBuffer();
});
const splatData = parseSplats(arrayBuffer);

const center = vec3.create();
for (const s of splatData) vec3.add(center, center, s.position);
vec3.scale(center, center, 1 / splatData.length);
const dtmp = new Float32Array(splatData.length);
for (let pass = 0; pass < 4; pass++) {
    for (let i = 0; i < splatData.length; i++) {
        const p = splatData[i].position;
        dtmp[i] = Math.hypot(p[0] - center[0], p[1] - center[1], p[2] - center[2]);
    }
    const cut = Float32Array.from(dtmp).sort()[Math.floor(dtmp.length * 0.6)] || Infinity;
    const acc = vec3.create();
    let count = 0;
    for (let i = 0; i < splatData.length; i++) {
        if (dtmp[i] <= cut) { vec3.add(acc, acc, splatData[i].position); count++; }
    }
    if (count) vec3.scale(center, acc, 1 / count);
}
const dists = new Float32Array(splatData.length);
for (let i = 0; i < splatData.length; i++) {
    const s = splatData[i];
    vec3.subtract(s.position, s.position, center);
    dists[i] = vec3.length(s.position);
}
const sorted = Float32Array.from(dists).sort();
const radius = sorted[Math.floor(sorted.length * 0.65)] || sorted[sorted.length - 1] || 1;
const distance = preset.dist ?? radius * 1.4;
log(`${splatData.length.toLocaleString()} splats loaded. Camera distance ${distance.toFixed(2)}.`);

// ---- Scene graph ----
const root = new Node();
const splatNode = new Node();
splatNode.addComponent(new Splat(splatData));
root.addChild(splatNode);
const cameraNode = new Node();
cameraNode.addComponent(new Transform());
cameraNode.addComponent(new Camera({ aspect: 1, fovy: 1, near: distance / 100, far: distance * 10 }));
root.addChild(cameraNode);

const TARGET = [preset.tx ?? 0, preset.ty ?? 0, preset.tz ?? 0];

// ---- Renderers + offscreen textures (as in capture.js) ----
const renderer = new SplatRenderer(device, 'rgba8unorm');
renderer.loBound = 0;
renderer.hiBound = 1;                                   // full stochastic noise (1 spp)
const accumulator = new Compositor(device, 'rgba32float'); // gamma = 1
const depthRenderer = new DepthRenderer(device);

const colorTexture = device.createTexture({
    size: [SIZE, SIZE], format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
});
const depthBuffer = device.createTexture({
    size: [SIZE, SIZE], format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
const accumTexture = device.createTexture({
    size: [SIZE, SIZE], format: 'rgba32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
});
const depthMRT = device.createTexture({
    size: [SIZE, SIZE], format: 'r32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
});
const colorTarget = { color: colorTexture, depth: depthBuffer };
const accumTarget = { color: accumTexture };
const depthTarget = { depth: depthMRT, depthBuffer: depthBuffer };

// ---- GPU readback (rgba32float = 16 B/px, r32float = 4 B/px) ----
async function readback(texture, bytesPerPixel) {
    const w = texture.width, h = texture.height;
    const padded = Math.ceil(w * bytesPerPixel / 256) * 256;
    const buffer = device.createBuffer({
        size: padded * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture },
        { buffer, bytesPerRow: padded, rowsPerImage: h }, { width: w, height: h });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const floatsPerPixel = bytesPerPixel / 4;
    const src = new Float32Array(buffer.getMappedRange());
    const floatsPerRow = padded / 4;
    const out = new Float32Array(w * h * floatsPerPixel);
    for (let y = 0; y < h; y++)
        for (let x = 0; x < w * floatsPerPixel; x++)
            out[y * w * floatsPerPixel + x] = src[y * floatsPerRow + x];
    buffer.unmap();
    buffer.destroy();
    return out;
}
const readRGBA = t => readback(t, 16);
const readDepth = t => readback(t, 4);

// ---- Depth normalization: reproduce dataset._normalize_depth exactly ----
// scale = 99th percentile of valid (>0) depths; out = clip(d/scale, 0, 1); bg = 0.
function percentile(sortedAsc, p) {
    const n = sortedAsc.length;
    if (n === 0) return 1;
    const rank = (p / 100) * (n - 1);
    const lo = Math.floor(rank), hi = Math.ceil(rank);
    return lo === hi ? sortedAsc[lo] : sortedAsc[lo] * (hi - rank) + sortedAsc[hi] * (rank - lo);
}
function normalizeDepth(depth, out) {
    const valid = [];
    for (let i = 0; i < depth.length; i++) if (depth[i] > 0) valid.push(depth[i]);
    if (valid.length === 0) { out.fill(0); return; }
    valid.sort((a, b) => a - b);
    let scale = percentile(valid, 99.0);
    if (scale <= 0) scale = valid[valid.length - 1];
    for (let i = 0; i < depth.length; i++) {
        const d = depth[i];
        out[i] = d > 0 ? Math.min(Math.max(d / scale, 0), 1) : 0;
    }
}

// ---- ORT-Web session (WebGPU EP, residual fallback handled by the model file) ----
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
log(`Loading denoiser model ${MODEL_URL} (onnxruntime-web, WebGPU)...`);
let session, ep = 'webgpu';
try {
    session = await ort.InferenceSession.create(MODEL_URL, {
        executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
    });
} catch (e) {
    log(`WebGPU EP failed (${e.message}); falling back to wasm (CPU).`);
    ep = 'wasm';
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
}
log(`Model ready on ${ep}. Inputs: ${session.inputNames}, outputs: ${session.outputNames}.`);

// Preallocated buffers reused every frame to avoid per-frame GC churn.
const HW = SIZE * SIZE;
const inputData = new Float32Array(4 * HW);   // NCHW [1,4,H,W]
const depthNorm = new Float32Array(HW);
const inputName = session.inputNames[0];
const outputName = session.outputNames[0];

// ---- Output canvases ----
const noisyCanvas = document.getElementById('noisy');
const denoisedCanvas = document.getElementById('denoised');
noisyCanvas.width = noisyCanvas.height = SIZE;
denoisedCanvas.width = denoisedCanvas.height = SIZE;
if (noisyCanvas.nextElementSibling)   // keep the caption truthful when SPP > 1
    noisyCanvas.nextElementSibling.textContent =
        `Renderer output — ${SPP} spp (noisy input)`;
const noisyCtx = noisyCanvas.getContext('2d');
const denoisedCtx = denoisedCanvas.getContext('2d');
const noisyImg = new ImageData(SIZE, SIZE);
const denoisedImg = new ImageData(SIZE, SIZE);
const cl255 = v => Math.max(0, Math.min(255, Math.round(v * 255)));

// ---- Orbit controls ----
let az = 0.6, el = 0.8, dist = distance, autoSpin = true;
function setCamera() {
    const ce = Math.cos(el), se = Math.sin(el);
    const eye = [
        TARGET[0] + dist * ce * Math.cos(az),
        TARGET[1] - dist * se,                 // +Y is down (COLMAP), orbit above
        TARGET[2] + dist * ce * Math.sin(az),
    ];
    cameraNode.getComponentOfType(Transform).matrix =
        mat4.targetTo(mat4.create(), eye, TARGET, [0, -1, 0]);
}
for (const cv of [noisyCanvas, denoisedCanvas]) {
    cv.addEventListener('pointerdown', e => { cv.setPointerCapture(e.pointerId); autoSpin = false; });
    cv.addEventListener('pointermove', e => {
        if (e.buttons === 0) return;
        az -= e.movementX * 0.008;
        el = Math.max(-1.4, Math.min(1.4, el + e.movementY * 0.008));
    });
    cv.addEventListener('wheel', e => {
        e.preventDefault();
        dist = Math.max(0.2, dist * Math.exp(e.deltaY * 0.001));
    }, { passive: false });
}
addEventListener('keydown', e => { if (e.key === ' ') { autoSpin = !autoSpin; e.preventDefault(); } });

// ---- Main loop ----
const nextFrame = () => new Promise(requestAnimationFrame);
let frames = 0, emaFps = 0, emaDenoise = 0;
log('Running. Drag to orbit.');

while (true) {
    const tFrame = performance.now();
    if (autoSpin) az += 0.004;
    setCamera();

    // Accumulate SPP stochastic passes into the running mean (1/i weights) to get
    // the noisy input at the requested sample count. SPP=1 is the maximally noisy,
    // real-time case the model was trained for; higher SPP is a cleaner (but
    // out-of-distribution) input and costs SPP render passes per displayed frame.
    for (let i = 1; i <= SPP; i++) {
        renderer.render(colorTarget, root, cameraNode);
        accumulator.render(accumTarget, colorTexture, 1 / i);
    }
    depthRenderer.render(depthTarget, root, cameraNode);

    const noisy = await readRGBA(accumTexture);   // [H*W*4] linear rgba in [0,1]
    const depth = await readDepth(depthMRT);      // [H*W] linear view-space depth
    normalizeDepth(depth, depthNorm);

    // Pack NCHW input: RGB from noisy, 4th channel = normalized depth.
    for (let i = 0; i < HW; i++) {
        inputData[0 * HW + i] = noisy[i * 4 + 0];
        inputData[1 * HW + i] = noisy[i * 4 + 1];
        inputData[2 * HW + i] = noisy[i * 4 + 2];
        inputData[3 * HW + i] = depthNorm[i];
    }

    const tInfer = performance.now();
    const feeds = { [inputName]: new ort.Tensor('float32', inputData, [1, 4, SIZE, SIZE]) };
    const out = (await session.run(feeds))[outputName].data;   // [1,3,H,W]
    const denoiseMs = performance.now() - tInfer;

    // Paint both canvases.
    for (let i = 0; i < HW; i++) {
        noisyImg.data[i * 4 + 0] = cl255(noisy[i * 4 + 0]);
        noisyImg.data[i * 4 + 1] = cl255(noisy[i * 4 + 1]);
        noisyImg.data[i * 4 + 2] = cl255(noisy[i * 4 + 2]);
        noisyImg.data[i * 4 + 3] = 255;
        denoisedImg.data[i * 4 + 0] = cl255(out[0 * HW + i]);
        denoisedImg.data[i * 4 + 1] = cl255(out[1 * HW + i]);
        denoisedImg.data[i * 4 + 2] = cl255(out[2 * HW + i]);
        denoisedImg.data[i * 4 + 3] = 255;
    }
    noisyCtx.putImageData(noisyImg, 0, 0);
    denoisedCtx.putImageData(denoisedImg, 0, 0);

    const frameMs = performance.now() - tFrame;
    emaFps = emaFps ? emaFps * 0.9 + (1000 / frameMs) * 0.1 : 1000 / frameMs;
    emaDenoise = emaDenoise ? emaDenoise * 0.9 + denoiseMs * 0.1 : denoiseMs;
    if (++frames % 4 === 0) {
        statsEl.textContent =
            `${SCENE} @ ${SIZE}²  |  input ${SPP} spp  |  ${emaFps.toFixed(1)} fps end-to-end  |  `
            + `denoise ${emaDenoise.toFixed(1)} ms (${ep})  |  scene render + readback included`;
    }
    await nextFrame();
}
