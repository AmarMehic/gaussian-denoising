export function parseSplats(arrayBuffer) {
    const splatByteLength = 32; // xyz (position, f32), xyz (scale, f32), rgba (u8), xyzw (rot, u8)
    const splatCount = Math.floor(arrayBuffer.byteLength / splatByteLength);

    const splats = [];

    const floats = new Float32Array(arrayBuffer);
    const bytes = new Uint8Array(arrayBuffer);

    for (let i = 0; i < splatCount; i++) {
        const position = [
            floats[8 * i + 0],
            floats[8 * i + 1],
            floats[8 * i + 2],
        ];

        const scale = [
            floats[8 * i + 3 + 0],
            floats[8 * i + 3 + 1],
            floats[8 * i + 3 + 2],
        ];

        const color = [
            bytes[32 * i + 24 + 0] / 255,
            bytes[32 * i + 24 + 1] / 255,
            bytes[32 * i + 24 + 2] / 255,
            bytes[32 * i + 24 + 3] / 255,
        ];

        const rotation = [
            (bytes[32 * i + 28 + 0] - 128) / 128,
            (bytes[32 * i + 28 + 1] - 128) / 128,
            (bytes[32 * i + 28 + 2] - 128) / 128,
            (bytes[32 * i + 28 + 3] - 128) / 128,
        ];

        splats.push({ position, scale, rotation, color });
    }

    return splats;
}
