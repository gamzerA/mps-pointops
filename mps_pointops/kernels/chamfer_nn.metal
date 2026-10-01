// SPDX-License-Identifier: Apache-2.0
#pragma METAL fp contract(off)
#include <metal_stdlib>

using namespace metal;

constant constexpr uint SIMD = 32;
constant constexpr uint QUERIES_PER_GROUP = 8;

// One SIMD group scans one query. The (distance, index) reduction chooses the
// first reference point when rounded squared distances tie. Each output row,
// including padding, is written by lane zero before the dispatch completes.
kernel void chamfer_nearest_f32(
    device const float *query [[buffer(0)]],
    device const float *ref [[buffer(1)]],
    device const long *query_lengths [[buffer(2)]],
    device const long *ref_lengths [[buffer(3)]],
    device float *out_distance_sq [[buffer(4)]],
    device long *out_index [[buffer(5)]],
    constant long &batch_count [[buffer(6)]],
    constant long &queries_per_batch [[buffer(7)]],
    constant long &refs_per_batch [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {

    const ulong row = ulong(group.x) * QUERIES_PER_GROUP + ulong(simdgroup);
    const ulong total = ulong(batch_count) * ulong(queries_per_batch);
    if (row >= total) {
        return;
    }
    const ulong batch = row / ulong(queries_per_batch);
    const ulong query_index = row % ulong(queries_per_batch);
    if (query_index >= ulong(query_lengths[batch])) {
        if (lane == 0) {
            out_distance_sq[row] = 0.0f;
            out_index[row] = -1;
        }
        return;
    }

    const ulong query_base = row * 3;
    const float qx = query[query_base];
    const float qy = query[query_base + 1];
    const float qz = query[query_base + 2];
    const ulong ref_base = batch * ulong(refs_per_batch) * 3;
    const uint valid_refs = uint(ref_lengths[batch]);
    float best_distance = INFINITY;
    uint best_index = UINT_MAX;
    for (ulong j = ulong(lane); j < ulong(valid_refs); j += SIMD) {
        const ulong offset = ref_base + j * 3;
        const float dx = qx - ref[offset];
        const float dy = qy - ref[offset + 1];
        const float dz = qz - ref[offset + 2];
        float distance = dx * dx;
        distance = distance + dy * dy;
        distance = distance + dz * dz;
        if (distance < best_distance ||
            (distance == best_distance && j < ulong(best_index))) {
            best_distance = distance;
            best_index = uint(j);
        }
    }

    const float nearest_distance = simd_min(best_distance);
    const uint nearest_index = simd_min(
        best_distance == nearest_distance ? best_index : UINT_MAX);
    if (lane == 0) {
        out_distance_sq[row] = nearest_distance;
        out_index[row] = long(nearest_index);
    }
}
