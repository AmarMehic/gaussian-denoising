import { GUI } from 'dat';
import { vec3, vec4, mat4, quat } from 'glm';

import { ResizeSystem } from 'engine/systems/ResizeSystem.js';
import { UpdateSystem } from 'engine/systems/UpdateSystem.js';

import {
    Camera,
    Node,
    Transform,
} from 'engine/core.js';

import { TouchController } from 'engine/controllers/TouchController.js';

import { parseSplats } from './parseSplats.js';
import { Splat } from './Splat.js';
import { SplatRenderer } from './SplatRenderer.js';
import { Compositor } from './Compositor.js';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice({ requiredFeatures: ['float32-blendable'] });
const canvas = document.querySelector('canvas');
const context = canvas.getContext('webgpu');
const format = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format });

const renderer = new SplatRenderer(device, format);
const compositorFloat = new Compositor(device, 'rgba32float');
const compositor = new Compositor(device, format);
compositor.gamma = 1;

let depthTexture, colorTexture, compositorTexture;
let nFrames = 0;

const scene = new Node();

const splatContainer = new Node();
scene.addChild(splatContainer);

const camera = new Node();
camera.addComponent(new Transform());
camera.addComponent(new Camera());
camera.addComponent(new TouchController(camera, canvas));
scene.addChild(camera);

canvas.addEventListener('dragover', e => {
    e.preventDefault();
});

canvas.addEventListener('drop', async e => {
    e.preventDefault();

    for (const child of splatContainer.children) {
        console.log('Removing existing splats');
        child.remove();
    }

    console.log('Adding new splats');
    console.log(e.dataTransfer.files);
    const arrayBuffers = await Promise.all([...e.dataTransfer.files].map(file => file.arrayBuffer()));
    for (const arrayBuffer of arrayBuffers) {
        console.log('New splat');
        const splatData = parseSplats(arrayBuffer);
        const splatMean = splatData
            .map(splat => splat.position)
            .reduce((a, p) => vec3.add(a, a, vec3.scale(vec3.create(), p, 1 / splatData.length)), vec3.create());
        for (const splat of splatData) {
            vec3.subtract(splat.position, splat.position, splatMean);
        }

        const splat = new Node();
        splat.addComponent(new Splat(splatData));
        splatContainer.addChild(splat);
        nFrames = 0;
    }
});

function update(t, dt) {
    scene.traverse(node => {
        for (const component of node.components) {
            component.update?.(t, dt);
        }
    });
}

camera.addComponent({
    lastTransform: camera.getComponentOfType(Transform).matrix,
    update() {
        const newTransform = camera.getComponentOfType(Transform).matrix;
        if (mat4.exactEquals(newTransform, this.lastTransform)) {
            return;
        }

        this.lastTransform = newTransform;
        nFrames = 0;
    }
});

function render() {
    if (!colorTexture || colorTexture.width !== canvas.width || colorTexture.height !== canvas.height) {
        colorTexture?.destroy();
        colorTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: format,
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    if (!depthTexture || depthTexture.width !== canvas.width || depthTexture.height !== canvas.height) {
        depthTexture?.destroy();
        depthTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    if (!compositorTexture || compositorTexture.width !== canvas.width || compositorTexture.height !== canvas.height) {
        compositorTexture?.destroy();
        compositorTexture = device.createTexture({
            size: [canvas.width, canvas.height],
            format: 'rgba32float',
            usage:
                GPUTextureUsage.RENDER_ATTACHMENT |
                GPUTextureUsage.TEXTURE_BINDING,
        });
    }

    const renderTarget = {
        color: colorTexture,
        depth: depthTexture,
    };

    const compositorTarget = {
        color: compositorTexture,
    };

    const canvasTarget = {
        color: context.getCurrentTexture(),
    };

    renderer.render(renderTarget, scene, camera);
    compositorFloat.render(compositorTarget, colorTexture, 1 / (++nFrames), 1);
    compositor.render(canvasTarget, compositorTexture);
}

function resize({ displaySize: { width, height }}) {
    camera.getComponentOfType(Camera).aspect = width / height;
    nFrames = 0;
}

new ResizeSystem({ canvas, resize }).start();
new UpdateSystem({ update, render }).start();

const gui = new GUI();
gui.add(renderer, 'splatScale', 0, 10);
gui.add(renderer, 'loBound', 0, 1);
gui.add(renderer, 'hiBound', 0, 1);
