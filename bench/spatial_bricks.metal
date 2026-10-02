// SPDX-License-Identifier: Apache-2.0
// Experimental fixed-size Morton bricks. Not a public operator or an LBVH.
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BRICK_SIZE = 128;
constant constexpr uint MAX_K = 32;

static inline bool before(float d, uint i, float e, uint j) {
    return d < e || (d == e && i < j);
}

static inline float squared_norm(float3 d) {
    float s = d.x * d.x;
    s = s + d.y * d.y;
    return s + d.z * d.z;
}

static inline float lower_bound_sq(float3 q, float3 lo, float3 hi) {
    const float3 d = max(max(lo - q, q - hi), float3(0.0f));
    return squared_norm(d);
}

kernel void spatial_brick_bounds_f32(
    device const float *points [[buffer(0)]],
    device const long *sorted_indices [[buffer(1)]],
    device float *bounds [[buffer(2)]],
    constant long &point_count [[buffer(3)]],
    constant long &brick_count [[buffer(4)]],
    uint brick [[thread_position_in_grid]]) {
    if (ulong(brick) >= ulong(brick_count)) return;
    float3 low = float3(INFINITY), high = float3(-INFINITY);
    const ulong start = ulong(brick) * BRICK_SIZE;
    const ulong end = min(start + BRICK_SIZE, ulong(point_count));
    for (ulong p = start; p < end; ++p) {
        const ulong source = ulong(sorted_indices[p]);
        const ulong off = source * 3;
        const float3 x = float3(points[off], points[off + 1], points[off + 2]);
        low = min(low, x);
        high = max(high, x);
    }
    const ulong off = ulong(brick) * 6;
    bounds[off] = low.x;
    bounds[off + 1] = low.y;
    bounds[off + 2] = low.z;
    bounds[off + 3] = high.x;
    bounds[off + 4] = high.y;
    bounds[off + 5] = high.z;
}

static inline void scan_brick(
    float3 q, uint brick, ulong point_count,
    device const float *points, device const long *sorted_indices,
    thread float *best_d, thread uint *best_i, uint k, thread uint &found) {
    const ulong begin = ulong(brick) * BRICK_SIZE;
    const ulong end = min(begin + BRICK_SIZE, point_count);
    for (ulong p = begin; p < end; ++p) {
        const uint source = uint(sorted_indices[p]);
        const ulong off = ulong(source) * 3;
        const float3 x = float3(points[off], points[off + 1], points[off + 2]);
        // Match flat_knn_indices: ref minus query, three separate squares
        // and two adds with contraction disabled at the compilation unit.
        const float d = squared_norm(x - q);
        if (!isfinite(d) || !before(d, source, best_d[k - 1], best_i[k - 1])) continue;
        uint position = 0;
        while (position < found && before(best_d[position], best_i[position], d, source)) ++position;
        if (position >= k) continue;
        const uint next = min(k, found + 1);
        for (uint s = next - 1; s > position; --s) {
            best_d[s] = best_d[s - 1];
            best_i[s] = best_i[s - 1];
        }
        best_d[position] = d;
        best_i[position] = source;
        found = next;
    }
}

kernel void spatial_brick_knn_f32(
    device const float *query [[buffer(0)]],
    device const float *points [[buffer(1)]],
    device const long *sorted_indices [[buffer(2)]],
    device const float *bounds [[buffer(3)]],
    device float *out_dist [[buffer(4)]],
    device long *out_idx [[buffer(5)]],
    constant long &query_count [[buffer(6)]],
    constant long &point_count [[buffer(7)]],
    constant long &brick_count [[buffer(8)]],
    constant long &K [[buffer(9)]],
    uint qi [[thread_position_in_grid]]) {
    if (ulong(qi) >= ulong(query_count)) return;
    const uint k = uint(K);
    const ulong qoff = ulong(qi) * 3;
    const float3 q = float3(query[qoff], query[qoff + 1], query[qoff + 2]);
    float best_d[MAX_K];
    uint best_i[MAX_K];
    for (uint j = 0; j < k; ++j) {
        best_d[j] = INFINITY;
        best_i[j] = UINT_MAX;
    }
    uint found = 0;
    if (all(isfinite(q)) && brick_count > 0) {
        // Seed near the closest brick box, then test all other boxes. The
        // second pass is exhaustive over AABBs; only a strict bound may
        // prune, so equal-distance candidates with smaller IDs survive.
        uint seed = 0;
        float seed_d = INFINITY;
        for (ulong b = 0; b < ulong(brick_count); ++b) {
            const ulong off = b * 6;
            const float3 lo = float3(bounds[off], bounds[off + 1], bounds[off + 2]);
            const float3 hi = float3(bounds[off + 3], bounds[off + 4], bounds[off + 5]);
            const float lb = lower_bound_sq(q, lo, hi);
            if (lb < seed_d) { seed = uint(b); seed_d = lb; }
        }
        scan_brick(q, seed, ulong(point_count), points, sorted_indices,
                   best_d, best_i, k, found);
        for (ulong b = 0; b < ulong(brick_count); ++b) {
            if (uint(b) == seed) continue;
            const ulong off = b * 6;
            const float3 lo = float3(bounds[off], bounds[off + 1], bounds[off + 2]);
            const float3 hi = float3(bounds[off + 3], bounds[off + 4], bounds[off + 5]);
            const float lb = lower_bound_sq(q, lo, hi);
            if (found == k && lb > best_d[k - 1]) continue;
            scan_brick(q, uint(b), ulong(point_count), points, sorted_indices,
                       best_d, best_i, k, found);
        }
    }
    const ulong row = ulong(qi) * ulong(k);
    for (uint j = 0; j < k; ++j) {
        out_dist[row + j] = best_d[j];
        out_idx[row + j] = best_i[j] == UINT_MAX ? -1 : long(best_i[j]);
    }
}
