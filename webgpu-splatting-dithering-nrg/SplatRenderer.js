import { vec3, mat4 } from 'glm';

import { Camera } from 'engine/core.js';

import {
    getLocalModelMatrix,
    getGlobalViewMatrix,
    getProjectionMatrix,
} from 'engine/core/SceneUtils.js';

import { parseFormat, createVertexBuffer } from 'engine/core/VertexUtils.js';

import { Splat } from './Splat.js';

const code = await fetch(new URL('SplatRenderer.wgsl', import.meta.url))
    .then(response => response.text());

export class SplatRenderer {

    constructor(device, format = 'rgba8unorm') {
        this.device = device;
        this.format = format;
        this.gpuObjects = new WeakMap();

        const module = this.device.createShaderModule({ code });

        this.instanceBufferLayout = {
            arrayStride: 48,
            stepMode: 'instance',
            attributes: [
                {
                    name: 'position',
                    shaderLocation: 1,
                    offset: 0,
                    format: 'float32x3',
                },
                {
                    name: 'color',
                    shaderLocation: 2,
                    offset: 12,
                    format: 'unorm8x4',
                },
                {
                    name: 'rotation',
                    shaderLocation: 3,
                    offset: 16,
                    format: 'float32x4',
                },
                {
                    name: 'scale',
                    shaderLocation: 4,
                    offset: 32,
                    format: 'float32x3',
                },
            ],
        };

        this.pipeline = this.device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module,
                buffers: [this.instanceBufferLayout],
            },
            fragment: {
                module,
                targets: [{ format: this.format }],
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
            primitive: {
                topology: 'triangle-strip',
            },
        });

        this.splatScale = 3;
        this.loBound = 0;
        this.hiBound = 1;
        this.gamma = 1;
    }

    prepareSplat(splat) {
        if (this.gpuObjects.has(splat)) {
            return this.gpuObjects.get(splat);
        }

        const instanceBufferArrayBuffer = createVertexBuffer(splat.splats, this.instanceBufferLayout);
        const instanceBuffer = this.device.createBuffer({
            size: instanceBufferArrayBuffer.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        });
        this.device.queue.writeBuffer(instanceBuffer, 0, instanceBufferArrayBuffer);

        const splatUniformBuffer = this.device.createBuffer({
            size: 96,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const splatBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(1),
            entries: [{
                binding: 0,
                resource: { buffer: splatUniformBuffer },
            }],
        });

        const gpuObjects = {
            instanceBuffer,
            instanceBufferArrayBuffer,
            splatUniformBuffer,
            splatBindGroup,
        };
        this.gpuObjects.set(splat, gpuObjects);
        return gpuObjects;
    }

    prepareCamera(camera) {
        if (this.gpuObjects.has(camera)) {
            return this.gpuObjects.get(camera);
        }

        const cameraUniformBuffer = this.device.createBuffer({
            size: 144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const cameraBindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{
                binding: 0,
                resource: { buffer: cameraUniformBuffer },
            }],
        });

        const gpuObjects = {
            cameraUniformBuffer,
            cameraBindGroup,
        };
        this.gpuObjects.set(camera, gpuObjects);
        return gpuObjects;
    }

    getCurrentDepthTexture() {
        const canvas = this.context.getCurrentTexture();
        if (!this.depthTexture || this.depthTexture.width !== canvas.width || this.depthTexture.height !== canvas.height) {
            this.depthTexture?.destroy();
            this.depthTexture = this.device.createTexture({
                size: [canvas.width, canvas.height],
                format: 'depth24plus',
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }
        return this.depthTexture;
    }

    render(renderTarget, scene, camera) {
        const commandEncoder = this.device.createCommandEncoder();
        this.renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: renderTarget.color.createView(),
                loadOp: 'clear',
                clearValue: [0, 0, 0, 1],
                storeOp: 'store',
            }],
            depthStencilAttachment: {
                view: renderTarget.depth.createView(),
                depthLoadOp: 'clear',
                depthClearValue: 1,
                depthStoreOp: 'discard',
            },
        });
        this.renderPass.setPipeline(this.pipeline);

        const viewMatrix = getGlobalViewMatrix(camera);
        const projectionMatrix = getProjectionMatrix(camera);
        const cameraComponent = camera.getComponentOfType(Camera);
        const { cameraUniformBuffer, cameraBindGroup } = this.prepareCamera(cameraComponent);
        this.device.queue.writeBuffer(cameraUniformBuffer, 0, viewMatrix);
        this.device.queue.writeBuffer(cameraUniformBuffer, 64, projectionMatrix);

        const minSplatSizeInPixels = 1;
        const screenResolutionInvSq = new Float32Array([
            minSplatSizeInPixels / renderTarget.color.width ** 2,
            minSplatSizeInPixels / renderTarget.color.height ** 2,
        ]);
        this.device.queue.writeBuffer(cameraUniformBuffer, 128, screenResolutionInvSq);
        this.renderPass.setBindGroup(0, cameraBindGroup);

        this.renderNode(scene);

        this.renderPass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

    renderNode(node, modelMatrix = mat4.create()) {
        const localMatrix = getLocalModelMatrix(node);
        modelMatrix = mat4.mul(mat4.create(), modelMatrix, localMatrix);

        const splats = node.getComponentsOfType(Splat);
        for (const splat of splats) {
            this.renderSplat(splat, modelMatrix);
        }

        for (const child of node.children) {
            this.renderNode(child, modelMatrix);
        }
    }

    renderSplat(splat, modelMatrix) {
        const { instanceBuffer, splatUniformBuffer, splatBindGroup } = this.prepareSplat(splat);
        this.device.queue.writeBuffer(splatUniformBuffer, 0, modelMatrix);
        this.device.queue.writeBuffer(splatUniformBuffer, 64, new Float32Array([
            this.splatScale,
            this.loBound,
            this.hiBound,
            performance.now(),
            this.gamma,
        ]));
        this.renderPass.setBindGroup(1, splatBindGroup);
        this.renderPass.setVertexBuffer(0, instanceBuffer);
        this.renderPass.draw(4, splat.splats.length);
    }

}
