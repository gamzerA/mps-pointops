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
    if (valid_refs == 0) {
        if (lane == 0) {
            out_distance_sq[row] = 0.0f;
            out_index[row] = -1;
        }
        return;
    }
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
        // Finite coordinates can still overflow squared distance to +inf.
        // Select the first valid candidate even then, so the saved index is
        // always safe for gather in the backward pass.
        if (best_index == UINT_MAX || distance < best_distance ||
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

// L1 uses the same first-minimum rule, but the ranking metric is the sum of
// absolute coordinate differences. It must not reuse squared-L2 indices.
kernel void chamfer_nearest_l1_f32(
    device const float *query [[buffer(0)]],
    device const float *ref [[buffer(1)]],
    device const long *query_lengths [[buffer(2)]],
    device const long *ref_lengths [[buffer(3)]],
    device float *out_distance [[buffer(4)]],
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
            out_distance[row] = 0.0f;
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
    if (valid_refs == 0) {
        if (lane == 0) {
            out_distance[row] = 0.0f;
            out_index[row] = -1;
        }
        return;
    }
    float best_distance = INFINITY;
    uint best_index = UINT_MAX;
    for (ulong j = ulong(lane); j < ulong(valid_refs); j += SIMD) {
        const ulong offset = ref_base + j * 3;
        const float dx = qx - ref[offset];
        const float dy = qy - ref[offset + 1];
        const float dz = qz - ref[offset + 2];
        float distance = fabs(dx);
        distance = distance + fabs(dy);
        distance = distance + fabs(dz);
        if (best_index == UINT_MAX || distance < best_distance ||
            (distance == best_distance && j < ulong(best_index))) {
            best_distance = distance;
            best_index = uint(j);
        }
    }

    const float nearest_distance = simd_min(best_distance);
    const uint nearest_index = simd_min(
        best_distance == nearest_distance ? best_index : UINT_MAX);
    if (lane == 0) {
        out_distance[row] = nearest_distance;
        out_index[row] = long(nearest_index);
    }
}

// Tensor-coordinate Chamfer for D != 3. The existing 3D kernels above keep
// their measured operation order. This path accumulates each coordinate in
// increasing dimension order, matching the pinned CPU kNN's scalar loop. It
// retains the first valid reference even when every finite-coordinate
// distance overflows to +infinity. Normals are handled only by the 3D path.
kernel void chamfer_nearest_nd_f32(
    device const float *query [[buffer(0)]],
    device const float *ref [[buffer(1)]],
    device const long *query_lengths [[buffer(2)]],
    device const long *ref_lengths [[buffer(3)]],
    device float *out_distance [[buffer(4)]],
    device long *out_index [[buffer(5)]],
    constant long &batch_count [[buffer(6)]],
    constant long &queries_per_batch [[buffer(7)]],
    constant long &refs_per_batch [[buffer(8)]],
    constant long &dimensions [[buffer(9)]],
    constant long &norm [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {

    const ulong row = ulong(group.x) * QUERIES_PER_GROUP + ulong(simdgroup);
    const ulong total = ulong(batch_count) * ulong(queries_per_batch);
    if (row >= total) return;

    const ulong batch = row / ulong(queries_per_batch);
    const ulong query_index = row % ulong(queries_per_batch);
    if (query_index >= ulong(query_lengths[batch])) {
        if (lane == 0) {
            out_distance[row] = 0.0f;
            out_index[row] = -1;
        }
        return;
    }

    const uint valid_refs = uint(ref_lengths[batch]);
    if (valid_refs == 0) {
        if (lane == 0) {
            out_distance[row] = 0.0f;
            out_index[row] = -1;
        }
        return;
    }

    const ulong dim = ulong(dimensions);
    const ulong query_base = row * dim;
    const ulong ref_base = batch * ulong(refs_per_batch) * dim;
    float best_distance = INFINITY;
    uint best_index = UINT_MAX;
    for (ulong j = ulong(lane); j < ulong(valid_refs); j += SIMD) {
        const ulong offset = ref_base + j * dim;
        const float first = query[query_base] - ref[offset];
        float distance = norm == 1 ? fabs(first) : first * first;
        for (ulong d = 1; d < dim; ++d) {
            const float delta = query[query_base + d] - ref[offset + d];
            const float term = norm == 1 ? fabs(delta) : delta * delta;
            distance = distance + term;
        }
        if (best_index == UINT_MAX || distance < best_distance ||
            (distance == best_distance && j < ulong(best_index))) {
            best_distance = distance;
            best_index = uint(j);
        }
    }

    const float nearest_distance = simd_min(best_distance);
    const uint nearest_index = simd_min(
        best_distance == nearest_distance ? best_index : UINT_MAX);
    if (lane == 0) {
        out_distance[row] = nearest_distance;
        out_index[row] = long(nearest_index);
    }
}
