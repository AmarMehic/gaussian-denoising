const vertices = array(
    vec2f(-1, -1),
    vec2f( 1, -1),
    vec2f(-1,  1),
    vec2f( 1,  1),
);

struct VertexInput {
    @builtin(vertex_index) vertexIndex: u32,
    @location(1) position: vec4f,
    @location(2) color: vec4f,
    @location(3) rotation: vec4f,
    @location(4) scale: vec3f,
}

struct VertexOutput {
    @builtin(position) clipPosition: vec4f,
    @location(0) position: vec2f,
    @location(1) color: vec4f,
}

struct FragmentInput {
    @location(0) position: vec2f,
    @location(1) color: vec4f,
}

struct FragmentOutput {
    @location(0) color: vec4f,
}

struct CameraUniforms {
    viewMatrix: mat4x4f,
    projectionMatrix: mat4x4f,
    screenResolutionInvSq: vec2f,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct SplatUniforms {
    modelMatrix: mat4x4f,
    scale: f32,
    loBound: f32,
    hiBound: f32,
    time: f32,
    gamma: f32,
}
@group(1) @binding(0) var<uniform> splat: SplatUniforms;

fn quaternionToMatrix(q: vec4f) -> mat3x3f {
    let x = q.y;
    let y = q.z;
    let z = q.w;
    let w = q.x;

    return mat3x3f(
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),

        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),

        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
    );
}

fn scaleToMatrix(s: vec3f) -> mat3x3f {
    return mat3x3f(
        s.x, 0, 0,
        0, s.y, 0,
        0, 0, s.z,
    );
}

fn projectionJacobian(p: vec3f) -> mat3x3f {
    let A = camera.projectionMatrix[0][0];
    let F = camera.projectionMatrix[1][1];
    let L = camera.projectionMatrix[3][2];
    let z2 = p.z * p.z;
    return mat3x3f(
        -A / p.z, 0, 0,
        0, -F / p.z, 0,
        A * p.x / z2, F * p.y / z2, L / z2,
    );
}

fn rand_f(state: ptr<function, u32>) -> f32 {
    *state = *state * 747796405u + 2891336453u;
    let word = ((*state >> ((*state >> 28u) + 4u)) ^ *state) * 277803737u;
    return f32((word >> 22u) ^ word) * bitcast<f32>(0x2f800004u);
}

fn h(x: u32) -> u32 {
    let y = x * 747796405u + 2891336453u;
    let z = ((y >> ((y >> 28u) + 4u)) ^ y) * 277803737u;
    return (z >> 22u) ^ z;
}

fn hash(in: vec3u) -> u32 {
    return h(h(h(in[0]) ^ in[1]) ^ in[2]);
}

@vertex
fn vertex(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let viewPosition = camera.viewMatrix * splat.modelMatrix * input.position;
    let screenPosition = camera.projectionMatrix * viewPosition;
    let screenPosition2D = screenPosition.xy / screenPosition.w;

    // Clip manually
    if (screenPosition.x > screenPosition.w || screenPosition.x < -screenPosition.w
        || screenPosition.y > screenPosition.w || screenPosition.y < -screenPosition.w) {
        output.clipPosition.z = 2;
        return output;
    }

    let R = quaternionToMatrix(input.rotation);
    let S = scaleToMatrix(input.scale);
    let V = mat3x3f(camera.viewMatrix[0].xyz, camera.viewMatrix[1].xyz, camera.viewMatrix[2].xyz);
    let J = projectionJacobian(viewPosition.xyz);

    let B = J * V * R * S;

    // Covariance matrix eigenvalues & eigenvectors
    let C = B * transpose(B);
    let a = C[0][0] + camera.screenResolutionInvSq.x;
    let b = C[0][1];
    let d = C[1][1] + camera.screenResolutionInvSq.y;
    let l = (a + d) / 2;
    let m = (a - d) / 2;
    let r = length(vec2f(m, b));
    let L1 = l + r;
    let L2 = l - r;
    let V1 = normalize(vec2f(b, L1 - a)) * sqrt(L1);
    let V2 = normalize(vec2f(a - L1, b)) * sqrt(L2);

    // 4 combinations for 4 vertices of the quad
    let vertex = vertices[input.vertexIndex];
    let x = vertex.x * V1 * splat.scale;
    let y = vertex.y * V2 * splat.scale;

    output.clipPosition = vec4f((screenPosition2D + x + y) * screenPosition.w, screenPosition.z, screenPosition.w);
    output.position = vertex * 2;
    output.color = input.color;

    return output;
}

@fragment
fn fragment(input: FragmentInput) -> FragmentOutput {
    var output: FragmentOutput;

    let distance2 = dot(input.position, input.position);
    if (distance2 > 4) {
        discard;
    }
    let alpha = exp(-distance2) * input.color.a;

    var state = hash(vec3u(bitcast<u32>(input.position.x), bitcast<u32>(input.position.y), bitcast<u32>(splat.time)));
    if (splat.loBound + (splat.hiBound - splat.loBound) * rand_f(&state) > alpha) {
        discard;
    }

    output.color = vec4f(input.color.rgb, 1);

    return output;
}
