@group(0) @binding(0) var colorTexture: texture_2d<f32>;
@group(0) @binding(1) var colorSampler: sampler;

struct Uniforms {
    alpha: f32,
    gamma: f32,
}

@group(0) @binding(2) var<uniform> uniforms: Uniforms;

const vertices = array(
    vec2f(-1, -1),
    vec2f( 3, -1),
    vec2f(-1,  3),
);

struct VertexInput {
    @builtin(vertex_index) index: u32,
}

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) texcoords: vec2f,
}

struct FragmentInput {
    @location(0) texcoords: vec2f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clipPosition = vec4f(vertices[input.index], 0, 1);
    output.texcoords = vertices[input.index] * vec2f(0.5, -0.5) + 0.5;
    return output;
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;
    let color = textureSample(colorTexture, colorSampler, input.texcoords);
    output.color = vec4f(pow(color.rgb, vec3f(uniforms.gamma)), uniforms.alpha);
    return output;
}
