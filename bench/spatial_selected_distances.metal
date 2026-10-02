// SPDX-License-Identifier: Apache-2.0
// Independent selected-pair distance probe; no spatial pruning or sort.
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

kernel void spatial_selected_sqdist_f32(
    device const float *query [[buffer(0)]],
    device const float *points [[buffer(1)]],
    device const long *indices [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant long &query_count [[buffer(4)]],
    constant long &K [[buffer(5)]],
    constant long &point_count [[buffer(6)]],
    uint linear [[thread_position_in_grid]]) {
    if (ulong(linear) >= ulong(query_count) * ulong(K)) return;
    const long source = indices[linear];
    if (source < 0 || source >= point_count) {
        out[linear] = INFINITY;
        return;
    }
    const ulong qoff = (ulong(linear) / ulong(K)) * 3;
    const ulong xoff = ulong(source) * 3;
    const float dx = points[xoff] - query[qoff];
    const float dy = points[xoff + 1] - query[qoff + 1];
    const float dz = points[xoff + 2] - query[qoff + 2];
    float d = dx * dx;
    d = d + dy * dy;
    out[linear] = d + dz * dz;
}
