import { mat4, vec3 } from 'glm';

import { Camera, Node, Transform } from 'engine/core.js';

import { parseSplats } from './parseSplats.js';
import { Splat } from './Splat.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';
import { DepthRenderer } from './DepthRenderer.js';

// ---- Parameters (override via URL) ----
// e.g. ?scene=bonsai-7k&n=60&spp=200&size=512&dist=14&elev=0.4,0.7,1.0&ty=0&preview=1
const params = new URLSearchParams(location.search);
const SCENE = params.get('scene') ?? 'bonsai-7k';

// Per-scene framing found by hand in interactive mode (?interactive=1, press P).
// URL params (tx/ty/tz/dist) still override these.
const SCENE_PRESETS = {
    'bonsai-7k': { tx: 2.7, ty: -2.4, tz: -2.11, dist: 5.0 },
    'bicycle-7k': { tx: 0.02, ty: 2.60, tz: 0.03, dist: 4.0 },
    'stump-7k': { tx: 0.38, ty: 4.17, tz: -1.63, dist: 4.4 },
    // New scenes (pivots from scripts/find_pivot.py; dist tuned in-browser).
    'counter-7k': { tx: -0.04, ty: -0.05, tz: 0.71, dist: 5.5 },
    'garden-7k': { tx: -1.61, ty: 0.43, tz: -1.11, dist: 4.6 },
    'kitchen-7k': { tx: 0.1, ty: 0.0, tz: 0.35, dist: 5.5 },
    // densest-voxel landed off the visual center here (sparse scene); the robust
    // scene-center origin orbits the room correctly instead.
    'playroom-7k': { tx: 0, ty: 0, tz: 0, dist: 6.0 },
    'room-7k': { tx: -2.64, ty: 0.49, tz: -0.1, dist: 5.0 },
};
const preset = SCENE_PRESETS[SCENE] ?? {};
const N_POSES = Number(params.get('n') ?? 80);
const SPP = Number(params.get('spp') ?? 400);       // frames accumulated for the clean target
// Noise-level snapshots of the running accumulator (spp counts), saved as
// separate _noisy<k>.png so the denoiser can train across noise strengths.
const NOISY_SPPS = (params.get('noisy') ?? '1,2,4').split(',').map(Number)
    .filter(k => k >= 1 && k <= SPP).sort((a, b) => a - b);
const SIZE = Number(params.get('size') ?? 512);     // square render resolution
const DIST = params.has('dist') ? Number(params.get('dist')) : (preset.dist ?? null); // camera distance
const ELEVS = (params.get('elev') ?? '0.6,0.8,1.0,1.2').split(',').map(Number); // look-down angles (rad)
const TARGET = [                                     // orbit pivot / look-at (after centering)
    Number(params.get('tx') ?? preset.tx ?? 0),
    Number(params.get('ty') ?? preset.ty ?? 0),
    Number(params.get('tz') ?? preset.tz ?? 0),
];
const PREVIEW_ONLY = params.get('preview') === '1'; // just orbit on-screen, don't save

const logEl = document.getElementById('log');
function log(msg) {
    logEl.textContent += '\n' + msg;
    console.log(msg);
}

// ---- Device ----
const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredFeatures: ['float32-blendable'] });

// ---- Load + center scene ----
log(`Loading scene "${SCENE}" ...`);
const arrayBuffer = await fetch(`../data/scenes/${SCENE}.splat`).then(r => {
    if (!r.ok) throw new Error(`Failed to fetch ${SCENE}.splat (${r.status})`);
    return r.arrayBuffer();
});
const splatData = parseSplats(arrayBuffer);

// Robust center: a plain centroid of a MipNeRF360 room is dragged off the
// subject by walls/floor/background. Start at the mean, then iteratively keep
// only the inner 60% of splats and recompute, so the center converges onto the
// densest cluster (the bonsai + table) -- that's what we orbit around.
const center = vec3.create();
for (const s of splatData) vec3.add(center, center, s.position);
vec3.scale(center, center, 1 / splatData.length);

const dtmp = new Float32Array(splatData.length);
for (let pass = 0; pass < 4; pass++) {
    for (let i = 0; i < splatData.length; i++) {
        const p = splatData[i].position;
        const dx = p[0] - center[0], dy = p[1] - center[1], dz = p[2] - center[2];
        dtmp[i] = Math.hypot(dx, dy, dz);
    }
    const cut = Float32Array.from(dtmp).sort()[Math.floor(dtmp.length * 0.6)] || Infinity;
    const acc = vec3.create();
    let count = 0;
    for (let i = 0; i < splatData.length; i++) {
        if (dtmp[i] <= cut) { vec3.add(acc, acc, splatData[i].position); count++; }
    }
    if (count) vec3.scale(center, acc, 1 / count);
}

// Center the scene on that point and measure the "content radius" from it
// (a percentile, not max, so unbounded background doesn't blow up the framing).
const dists = new Float32Array(splatData.length);
for (let i = 0; i < splatData.length; i++) {
    const s = splatData[i];
    vec3.subtract(s.position, s.position, center);
    dists[i] = vec3.length(s.position);
}
const sorted = Float32Array.from(dists).sort();
const radius = sorted[Math.floor(sorted.length * 0.65)] || sorted[sorted.length - 1] || 1;
const distance = DIST ?? radius * 1.4;
log(`${splatData.length.toLocaleString()} splats, content radius (p65) ${radius.toFixed(2)}, camera distance ${distance.toFixed(2)}`);

// ---- Scene graph ----
const root = new Node();
const splatNode = new Node();
splatNode.addComponent(new Splat(splatData));
root.addChild(splatNode);

const cameraNode = new Node();
cameraNode.addComponent(new Transform());
cameraNode.addComponent(new Camera({
    aspect: 1,
    fovy: 1,
    near: distance / 100,
    far: distance * 10,
}));
root.addChild(cameraNode);

// ---- Renderers ----
const renderer = new SplatRenderer(device, 'rgba8unorm');
renderer.loBound = 0;
renderer.hiBound = 1; // full stochastic range -> maximum single-frame noise
const accumulator = new Compositor(device, 'rgba32float'); // gamma defaults to 1
const depthRenderer = new DepthRenderer(device);

// ---- Offscreen textures ----
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

// ---- Camera poses: orbit rings at a few (look-down) elevations ----
// The scene is a flat horizontal slab (up = +y), so positive elevations look
// down onto it; elevation 0 would be edge-on and useless.
function makePoses(n) {
    const perRing = Math.ceil(n / ELEVS.length);
    const poses = [];
    for (const elev of ELEVS) {
        const ce = Math.cos(elev), se = Math.sin(elev);
        for (let k = 0; k < perRing; k++) {
            const az = 2 * Math.PI * k / perRing;
            poses.push([
                TARGET[0] + distance * ce * Math.cos(az),
                TARGET[1] - distance * se,   // +Y is down (COLMAP); subtract to orbit above
                TARGET[2] + distance * ce * Math.sin(az),
            ]);
        }
    }
    return poses.slice(0, n);
}
const poses = makePoses(N_POSES);

function setCamera(eye) {
    const transform = cameraNode.getComponentOfType(Transform);
    transform.matrix = mat4.targetTo(mat4.create(), eye, TARGET, [0, -1, 0]);
}

// ---- On-screen preview (WebGPU) so framing is visible while capturing ----
const canvas = document.querySelector('canvas');
canvas.width = canvas.height = SIZE;
const canvasContext = canvas.getContext('webgpu');
const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
canvasContext.configure({ device, format: canvasFormat });
const present = new Compositor(device, canvasFormat);
present.gamma = 1;
function showOnCanvas() {
    present.render({ color: canvasContext.getCurrentTexture() }, accumTexture);
}

// ---- GPU readback helpers ----
async function readback(texture, bytesPerPixel) {
    const w = texture.width, h = texture.height;
    const unpadded = w * bytesPerPixel;
    const padded = Math.ceil(unpadded / 256) * 256;
    const buffer = device.createBuffer({
        size: padded * h,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer(
        { texture },
        { buffer, bytesPerRow: padded, rowsPerImage: h },
        { width: w, height: h },
    );
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);

    const floatsPerPixel = bytesPerPixel / 4;
    const src = new Float32Array(buffer.getMappedRange());
    const floatsPerRow = padded / 4;
    const out = new Float32Array(w * h * floatsPerPixel);
    for (let y = 0; y < h; y++) {
        for (let x = 0; x < w * floatsPerPixel; x++) {
            out[y * w * floatsPerPixel + x] = src[y * floatsPerRow + x];
        }
    }
    buffer.unmap();
    buffer.destroy();
    return out;
}

const readRGBA = texture => readback(texture, 16); // rgba32float
const readDepth = texture => readback(texture, 4); // r32float

// ---- Encoding ----
function clamp255(v) { return Math.max(0, Math.min(255, Math.round(v * 255))); }

function rgbaToBlob(float32, w, h) {
    const img = new ImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        img.data[i * 4 + 0] = clamp255(float32[i * 4 + 0]);
        img.data[i * 4 + 1] = clamp255(float32[i * 4 + 1]);
        img.data[i * 4 + 2] = clamp255(float32[i * 4 + 2]);
        img.data[i * 4 + 3] = 255;
    }
    const canvas = new OffscreenCanvas(w, h);
    canvas.getContext('2d').putImageData(img, 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
}

function depthToBlob(float32, w, h) {
    // Normalize visible (nonzero) depth to 0..255 for a viewable preview.
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < w * h; i++) {
        const d = float32[i];
        if (d > 0) { lo = Math.min(lo, d); hi = Math.max(hi, d); }
    }
    const range = hi > lo ? hi - lo : 1;
    const img = new ImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        const d = float32[i];
        const v = d > 0 ? Math.round(255 * (1 - (d - lo) / range)) : 0; // near = bright
        img.data[i * 4 + 0] = v;
        img.data[i * 4 + 1] = v;
        img.data[i * 4 + 2] = v;
        img.data[i * 4 + 3] = 255;
    }
    const canvas = new OffscreenCanvas(w, h);
    canvas.getContext('2d').putImageData(img, 0, 0);
    return canvas.convertToBlob({ type: 'image/png' });
}

async function save(relPath, body) {
    const res = await fetch('/save?path=' + encodeURIComponent(relPath), {
        method: 'POST',
        body,
    });
    if (!res.ok) throw new Error(`save failed for ${relPath}: ${res.status}`);
}

const nextFrame = () => new Promise(requestAnimationFrame);

// Background-safe yield: requestAnimationFrame is paused when the tab is hidden
// or occluded (e.g. while a capture is driven via automation), which would stall
// the capture loop forever. A MessageChannel ping is delivered even in background
// tabs and isn't clamped like setTimeout, so the capture keeps running headless.
const yieldChannel = new MessageChannel();
const yieldQueue = [];
yieldChannel.port1.onmessage = () => yieldQueue.shift()?.();
const yieldToLoop = () => new Promise(res => {
    yieldQueue.push(res);
    yieldChannel.port2.postMessage(0);
});

// ---- Interactive mode: fly the camera by hand to find the subject center ----
// drag = orbit, wheel = zoom, WASD = slide the look-at on the ground, R/F = up/down,
// P = print copy-paste capture params. The image refines (denoises) whenever you
// hold still and resets the moment you move.
if (params.get('interactive') === '1') {
    let az = 0.6, el = 0.6, dist = distance;
    const target = [...TARGET];
    let acc = 0;
    const reset = () => { acc = 0; };

    const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    const norm = v => { const m = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / m, v[1] / m, v[2] / m]; };
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const eyeOf = () => {
        const ce = Math.cos(el), se = Math.sin(el);
        return [
            target[0] + dist * ce * Math.cos(az),
            target[1] + dist * se,
            target[2] + dist * ce * Math.sin(az),
        ];
    };

    // Click on the bonsai -> read the depth at that pixel, unproject to a 3D
    // world point, and make it the new orbit pivot. This puts the pivot on the
    // actual surface instead of in empty space.
    async function setPivotFromClick(px, py) {
        const eye = eyeOf();
        cameraNode.getComponentOfType(Transform).matrix =
            mat4.targetTo(mat4.create(), eye, target, [0, 1, 0]);
        depthRenderer.render(depthTarget, root, cameraNode);
        const buf = await readDepth(depthMRT);
        const d = buf[py * SIZE + px];
        if (!(d > 0)) { log('Clicked empty space (no geometry there) — try clicking on the plant.'); return; }
        const fwd = norm(sub(target, eye));
        const right = norm(cross(fwd, [0, 1, 0]));
        const up = cross(right, fwd);
        const tanHalf = Math.tan(0.5);            // fovy = 1 rad
        const ndcx = 2 * px / SIZE - 1;
        const ndcy = 1 - 2 * py / SIZE;
        for (let i = 0; i < 3; i++) {
            target[i] = eye[i] + fwd[i] * d + right[i] * ndcx * tanHalf * d + up[i] * ndcy * tanHalf * d;
        }
        dist = Math.hypot(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]);
        log(`Pivot -> [${target[0].toFixed(2)}, ${target[1].toFixed(2)}, ${target[2].toFixed(2)}] (depth ${d.toFixed(2)}). Orbit with arrows to verify it stays centered.`);
        reset();
    }

    let downX = 0, downY = 0, moved = false;
    canvas.addEventListener('pointerdown', e => {
        downX = e.clientX; downY = e.clientY; moved = false;
        canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', e => {
        if (e.buttons === 0) return;
        if (Math.abs(e.clientX - downX) + Math.abs(e.clientY - downY) > 4) moved = true;
        az -= e.movementX * 0.008;
        el = Math.max(-1.5, Math.min(1.5, el + e.movementY * 0.008));
        reset();
    });
    canvas.addEventListener('pointerup', e => {
        if (moved) return;                        // it was a drag, not a click
        const r = canvas.getBoundingClientRect();
        const px = Math.floor((e.clientX - r.left) * SIZE / r.width);
        const py = Math.floor((e.clientY - r.top) * SIZE / r.height);
        setPivotFromClick(px, py);
    });
    canvas.addEventListener('wheel', e => {
        e.preventDefault();
        dist = Math.max(0.05, dist * Math.exp(e.deltaY * 0.001));
        reset();
    }, { passive: false });
    addEventListener('keydown', e => {
        const step = dist * 0.03;
        const fwd = [-Math.cos(az), 0, -Math.sin(az)];   // ground forward
        const right = [Math.sin(az), 0, -Math.cos(az)];  // ground right
        const k = e.key.toLowerCase();
        // Orbit with arrow keys (no mouse needed).
        if (k === 'arrowleft') az -= 0.08;
        else if (k === 'arrowright') az += 0.08;
        else if (k === 'arrowup') el = Math.min(1.5, el + 0.06);
        else if (k === 'arrowdown') el = Math.max(-1.5, el - 0.06);
        // Zoom with Q/E.
        else if (k === 'q') dist = Math.min(dist * 1.05, 1e6);
        else if (k === 'e') dist = Math.max(0.05, dist * 0.95);
        // Slide the look-at center with WASD, R/F for up/down.
        else if (k === 'w') { target[0] += fwd[0] * step; target[2] += fwd[2] * step; }
        else if (k === 's') { target[0] -= fwd[0] * step; target[2] -= fwd[2] * step; }
        else if (k === 'd') { target[0] += right[0] * step; target[2] += right[2] * step; }
        else if (k === 'a') { target[0] -= right[0] * step; target[2] -= right[2] * step; }
        else if (k === 'r') target[1] += step;
        else if (k === 'f') target[1] -= step;
        else if (k === ' ' || k === 'enter') { setPivotFromClick(SIZE >> 1, SIZE >> 1); e.preventDefault(); return; }
        else if (k === 'p') {
            log(`-> &tx=${target[0].toFixed(2)}&ty=${target[1].toFixed(2)}&tz=${target[2].toFixed(2)}&dist=${dist.toFixed(2)}`);
            return;
        } else return;
        e.preventDefault();
        reset();
    });

    log('Interactive: aim the bonsai under the RED DOT, press SPACE to set pivot.');
    log('arrows=orbit, Q/E=zoom, drag=orbit, wheel=zoom, click=pivot too, P=print params.');

    // Fixed red crosshair at the canvas center as an aiming reticle.
    const dot = document.createElement('div');
    dot.style.cssText = 'position:fixed;width:10px;height:10px;margin:-5px 0 0 -5px;'
        + 'border-radius:50%;background:red;box-shadow:0 0 0 1px #fff;pointer-events:none;z-index:10;';
    document.body.appendChild(dot);
    const placeDot = () => {
        const r = canvas.getBoundingClientRect();
        dot.style.left = (r.left + r.width / 2) + 'px';
        dot.style.top = (r.top + r.height / 2) + 'px';
    };
    addEventListener('resize', placeDot);
    addEventListener('scroll', placeDot, { passive: true });

    while (true) {
        placeDot();
        const ce = Math.cos(el), se = Math.sin(el);
        const eye = [
            target[0] + dist * ce * Math.cos(az),
            target[1] + dist * se,
            target[2] + dist * ce * Math.sin(az),
        ];
        cameraNode.getComponentOfType(Transform).matrix =
            mat4.targetTo(mat4.create(), eye, target, [0, 1, 0]);
        acc++;
        renderer.render(colorTarget, root, cameraNode);
        accumulator.render(accumTarget, colorTexture, 1 / acc);
        showOnCanvas();
        await nextFrame();
    }
}

// ---- Preview-only mode: converge each pose fully on screen, then hold ----
if (PREVIEW_ONLY) {
    const previewSpp = Number(params.get('spp') ?? 160);
    const fixedPose = params.has('pose') ? Number(params.get('pose')) : null;
    log(`Preview mode (nothing saved). ${previewSpp} spp per pose.`);
    log(fixedPose !== null
        ? `Showing pose ${fixedPose} only.`
        : `Cycling ${poses.length} poses; hold to study each.`);
    log(`Tune with ?dist=, ?elev=, ?ty=, ?pose=, then drop &preview=1 to capture.`);
    let p = fixedPose ?? 0;
    while (true) {
        setCamera(poses[p % poses.length]);
        // Accumulate to a clean image, updating the canvas as it converges.
        for (let i = 1; i <= previewSpp; i++) {
            renderer.render(colorTarget, root, cameraNode);
            accumulator.render(accumTarget, colorTexture, 1 / i);
            if (i % 8 === 0) { showOnCanvas(); await nextFrame(); }
        }
        showOnCanvas();
        // Hold the converged image still (~1.5s) so it's actually readable.
        for (let h = 0; h < 90; h++) await nextFrame();
        if (fixedPose === null) p++;
    }
}

// ---- Capture loop ----
log(`Capturing ${poses.length} poses @ ${SIZE}x${SIZE}: clean ${SPP} spp, noisy [${NOISY_SPPS.join(',')}] spp -> data/renders/${SCENE}/`);
const t0 = performance.now();

for (let p = 0; p < poses.length; p++) {
    setCamera(poses[p]);

    const noisyLevels = {};                  // spp -> readback at that accumulation count
    const noisySet = new Set(NOISY_SPPS);
    for (let i = 1; i <= SPP; i++) {
        renderer.render(colorTarget, root, cameraNode);
        accumulator.render(accumTarget, colorTexture, 1 / i);
        if (noisySet.has(i)) noisyLevels[i] = await readRGBA(accumTexture);
        if (i % 32 === 0) { showOnCanvas(); await yieldToLoop(); } // live feedback
    }
    const clean = await readRGBA(accumTexture);
    showOnCanvas();

    depthRenderer.render(depthTarget, root, cameraNode);
    const depth = await readDepth(depthMRT);

    const id = String(p).padStart(4, '0');
    const base = `${SCENE}/${id}`;
    for (const k of NOISY_SPPS) {
        await save(`${base}_noisy${k}.png`, await rgbaToBlob(noisyLevels[k], SIZE, SIZE));
    }
    await save(`${base}_clean.png`, await rgbaToBlob(clean, SIZE, SIZE));
    await save(`${base}_depth.png`, await depthToBlob(depth, SIZE, SIZE));
    await save(`${base}_depth.f32`, depth.buffer); // raw linear depth, row-major

    const elapsed = (performance.now() - t0) / 1000;
    log(`[${p + 1}/${poses.length}] ${base}  (${elapsed.toFixed(1)}s)`);
}

log(`Done. ${poses.length} triplets written in ${((performance.now() - t0) / 1000).toFixed(1)}s.`);
