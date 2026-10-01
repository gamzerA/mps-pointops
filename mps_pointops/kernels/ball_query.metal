// SPDX-License-Identifier: MIT
#pragma METAL fp contract(off)
#include <metal_stdlib>

using namespace metal;

constant constexpr uint SIMD = 32;
constant constexpr uint QUERIES_PER_GROUP = 8;

// Keep the three-coordinate accumulation order explicit. The first product
// rounds separately; the following products use an explicit fused multiply-add.
// Metal may still flush subnormal operands/results on the device.
inline float squared_norm(float3 delta) {
    float sum = delta.x * delta.x;
    sum = fma(delta.y, delta.y, sum);
    return fma(delta.z, delta.z, sum);
}

// The host validates buffer sizes and the products B*Q, B*P*3, and B*Q*K
// before dispatch. One SIMD group owns a query. Consecutive 32-point blocks
// and an exclusive rank within each block preserve the first-K source order.
template <typename Scalar>
inline float3 load_xyz(device const Scalar *coordinates, ulong offset) {
    return float3(float(coordinates[offset]),
                  float(coordinates[offset + 1]),
                  float(coordinates[offset + 2]));
}

template <typename Scalar>
inline void ball_query_impl(
    device const Scalar *queries,
    device const Scalar *points,
    device const long *lengths1,
    device const long *lengths2,
    device long *indices,
    device float *dists,
    constant long &batch_count,
    constant long &queries_per_batch,
    constant long &points_per_batch,
    constant long &max_neighbors,
    constant float &radius_sq,
    constant float &radius,
    uint3 group,
    uint lane,
    uint simdgroup) {

    if (batch_count <= 0 || queries_per_batch <= 0 ||
        points_per_batch < 0 || max_neighbors <= 0) {
        return;
    }

    const ulong query_index = ulong(group.x) * QUERIES_PER_GROUP + ulong(simdgroup);
    const ulong q_count = ulong(queries_per_batch);
    const ulong batch = query_index / q_count;
    if (batch >= ulong(batch_count)) {
        return;
    }
    const ulong query_in_batch = query_index % q_count;
    const ulong k_count = ulong(max_neighbors);

    // Each hit has a unique SIMD prefix rank; remaining slots are padded by
    // separate lanes after the scan. No atomics or cross-group ordering needed.
    const ulong output_base = query_index * k_count;
    const long query_length = lengths1[batch];
    const long point_length = lengths2[batch];
    ulong found = 0;
    if (query_length > 0 && point_length > 0 && points_per_batch > 0 &&
        query_in_batch < min(ulong(query_length), q_count) &&
        radius > 0.0f) {
        // The host checks B*P*3 against the allocated buffer size and 64-bit range.
        const ulong p_count = ulong(points_per_batch);
        const ulong point_base = batch * p_count * 3;
        const float3 query = load_xyz(queries, query_index * 3);
        if (all(isfinite(query))) {
            const ulong valid_points = min(ulong(point_length), p_count);
            for (ulong base = 0; base < valid_points; base += SIMD) {
                const ulong point_index = base + ulong(lane);
                bool within_radius = false;
                float distance_sq = 0.0f;
                if (point_index < valid_points) {
                    const float3 point = load_xyz(points, point_base + point_index * 3);
                    if (all(isfinite(point))) {
                        const float3 delta = query - point;
                        if (all(isfinite(delta))) {
                            // Even with a normal radius square, one component
                            // square can be subnormal and flush to zero. For
                            // small radii, normalize before comparing.
                            if (radius < 0x1p-50f || !isfinite(radius_sq)) {
                                const float normalized_sq = squared_norm(delta / radius);
                                within_radius = normalized_sq < 1.0f;
                                distance_sq = within_radius
                                    ? ((radius_sq >= 0x1p-126f && isfinite(radius_sq))
                                        ? normalized_sq * radius_sq
                                        : squared_norm(delta))
                                    : 0.0f;
                            } else {
                                distance_sq = squared_norm(delta);
                                within_radius = distance_sq < radius_sq;
                            }
                        }
                    }
                }

                // Every lane participates, including out-of-range lanes in
                // the final block. The exclusive rank gives original j order.
                const uint rank = simd_prefix_exclusive_sum(uint(within_radius));
                const uint block_hits = simd_sum(uint(within_radius));
                if (within_radius && found + ulong(rank) < k_count) {
                    const ulong slot = output_base + found + ulong(rank);
                    indices[slot] = long(point_index);
                    dists[slot] = distance_sq;
                }
                found = min(k_count, found + ulong(block_hits));
                if (found == k_count) {
                    break;
                }
            }
        }
    }
    for (ulong slot = found + ulong(lane); slot < k_count; slot += SIMD) {
        indices[output_base + slot] = -1;
        dists[output_base + slot] = 0.0f;
    }
}

kernel void ball_query_f32(
    device const float *queries [[buffer(0)]],
    device const float *points [[buffer(1)]],
    device const long *lengths1 [[buffer(2)]],
    device const long *lengths2 [[buffer(3)]],
    device long *indices [[buffer(4)]],
    device float *dists [[buffer(5)]],
    constant long &batch_count [[buffer(6)]],
    constant long &queries_per_batch [[buffer(7)]],
    constant long &points_per_batch [[buffer(8)]],
    constant long &max_neighbors [[buffer(9)]],
    constant float &radius_sq [[buffer(10)]],
    constant float &radius [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    ball_query_impl(queries, points, lengths1, lengths2, indices, dists,
                    batch_count, queries_per_batch, points_per_batch,
                    max_neighbors, radius_sq, radius, group, lane, simdgroup);
}

kernel void ball_query_f16(
    device const half *queries [[buffer(0)]],
    device const half *points [[buffer(1)]],
    device const long *lengths1 [[buffer(2)]],
    device const long *lengths2 [[buffer(3)]],
    device long *indices [[buffer(4)]],
    device float *dists [[buffer(5)]],
    constant long &batch_count [[buffer(6)]],
    constant long &queries_per_batch [[buffer(7)]],
    constant long &points_per_batch [[buffer(8)]],
    constant long &max_neighbors [[buffer(9)]],
    constant float &radius_sq [[buffer(10)]],
    constant float &radius [[buffer(11)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    ball_query_impl(queries, points, lengths1, lengths2, indices, dists,
                    batch_count, queries_per_batch, points_per_batch,
                    max_neighbors, radius_sq, radius, group, lane, simdgroup);
}
