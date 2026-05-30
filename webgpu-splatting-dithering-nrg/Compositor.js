const code = await fetch(new URL('Compositor.wgsl', import.meta.url))
    .then(response => response.text());

export class Compositor {

    constructor(device, format = 'rgba8unorm') {
        this.device = device;
        this.format = format;

        this.layout = this.device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: { sampleType: 'unfilterable-float' },
                },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'non-filtering'} },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: {} },
            ]
        });

        const module = this.device.createShaderModule({ code });
        this.pipeline = this.device.createRenderPipeline({
            //layout: 'auto',
            layout: this.device.createPipelineLayout({
                bindGroupLayouts: [this.layout],
            }),
            vertex: {
                module,
            },
            fragment: {
                module,
                targets: [{
                    format: this.format,
                    blend: {
                        color: {
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha',
                        },
                        alpha: {
                            srcFactor: 'zero',
                            dstFactor: 'one',
                        },
                    },
                }],
            },
        });

        this.sampler = this.device.createSampler({

        });

        this.uniformBuffer = this.device.createBuffer({
            size: 8,
            usage:
                GPUBufferUsage.UNIFORM |
                GPUBufferUsage.COPY_DST,
        });

        this.gamma = 1;
    }

    render(renderTarget, texture, alpha = 1) {
        this.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array([alpha, this.gamma]));

        const bindGroup = this.device.createBindGroup({
            //layout: this.pipeline.getBindGroupLayout(0),
            layout: this.layout,
            entries: [
                { binding: 0, resource: texture.createView() },
                { binding: 1, resource: this.sampler },
                { binding: 2, resource: this.uniformBuffer },
            ],
        });

        const commandEncoder = this.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: renderTarget.color.createView(),
                loadOp: 'load',
                storeOp: 'store',
            }],
        });
        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, bindGroup);
        renderPass.draw(3);
        renderPass.end();
        this.device.queue.submit([commandEncoder.finish()]);
    }

}
