// SPDX-License-Identifier: Apache-2.0
// Experimental v0.9 stage 1: 21-bit-per-axis Morton keys. No public API.
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

static inline ulong spread_21(uint value) {
    ulong bits = ulong(value & 0x1fffffu);
    bits = (bits | (bits << 32)) & 0x1f00000000fffful;
    bits = (bits | (bits << 16)) & 0x1f0000ff0000fful;
    bits = (bits | (bits << 8)) & 0x100f00f00f00f00ful;
    bits = (bits | (bits << 4)) & 0x10c30c30c30c30c3ul;
    return (bits | (bits << 2)) & 0x1249249249249249ul;
}

kernel void spatial_morton_keys_f32(
    device const float *points [[buffer(0)]],
    device const float *origin [[buffer(1)]],
    device long *keys [[buffer(2)]],
    device uchar *invalid [[buffer(3)]],
    constant long &count [[buffer(4)]],
    constant float &cell_size [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (ulong(i) >= ulong(count)) return;
    const ulong offset = ulong(i) * 3;
    const float3 point = float3(points[offset], points[offset + 1], points[offset + 2]);
    const float3 base = float3(origin[0], origin[1], origin[2]);
    // The precise floating-point quantization policy is part of this stage's
    // contract. Later query traversal must conservatively cover boundary
    // rounding. Never wrap a cell coordinate into a seemingly valid key.
    const float3 cell = floor((point - base) / cell_size);
    const bool good = all(isfinite(point)) && all(isfinite(base)) &&
                      all(isfinite(cell)) && all(cell >= 0.0f) &&
                      all(cell < 2097152.0f);
    if (!good) {
        keys[i] = LONG_MAX;
        invalid[i] = uchar(1);
        return;
    }
    const uint cx = uint(cell.x);
    const uint cy = uint(cell.y);
    const uint cz = uint(cell.z);
    keys[i] = long(spread_21(cx) | (spread_21(cy) << 1) | (spread_21(cz) << 2));
    invalid[i] = uchar(0);
}
