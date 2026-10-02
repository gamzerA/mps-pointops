// SPDX-License-Identifier: Apache-2.0
// The radius arithmetic below adapts this project's MIT-licensed Ball Query
// numerical helper. Its preserved notice is LICENSES/MIT-ball-query.txt.
#include <metal_stdlib>
using namespace metal;

// Flat point clouds are grouped by nondecreasing batch vectors. ptr_x and
// ptr_y contain the start of each batch and a final total length. One
// simdgroup owns a query for both searches. Radius scans reference points in
// consecutive 32-point blocks and uses a SIMD prefix to preserve first-K.

constant constexpr uint SIMD = 32;
constant constexpr uint QUERIES_PER_GROUP = 8;
constant constexpr uint MAX_K = 256;
constant constexpr uint SLOTS = MAX_K / SIMD;

static inline uint batch_for_query(device const long* ptr_y, uint batch_count, ulong query) {
    uint lo = 0;
    uint hi = batch_count;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        if (ulong(ptr_y[mid + 1]) <= query) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static inline bool before(float d, uint i, float e, uint j) {
    return d < e || (d == e && i < j);
}

static inline uint gcd_uint(uint a, uint b) {
    while (b != 0) {
        const uint next = a % b;
        a = b;
        b = next;
    }
    return a;
}

kernel void flat_knn_indices(
    device const float* query [[buffer(0)]],
    device const float* ref [[buffer(1)]],
    device const long* ptr_y [[buffer(2)]],
    device const long* ptr_x [[buffer(3)]],
    device long* out_idx [[buffer(4)]],
    constant long& query_count [[buffer(5)]],
    constant long& batch_count [[buffer(6)]],
    constant long& K [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    #pragma clang fp contract(off)
    threadgroup float list_d[QUERIES_PER_GROUP][MAX_K];
    threadgroup uint list_i[QUERIES_PER_GROUP][MAX_K];

    const ulong query_index = ulong(group.x) * QUERIES_PER_GROUP + ulong(sg);
    if (query_index >= ulong(query_count)) {
        return;
    }
    const uint q = uint(query_index);
    const uint b = batch_for_query(ptr_y, uint(batch_count), ulong(q));
    const ulong x_begin = ulong(ptr_x[b]);
    const ulong n_long = ulong(ptr_x[b + 1]) - x_begin;
    const uint n = uint(n_long);
    const uint k = uint(K);
    threadgroup float* ld = list_d[sg];
    threadgroup uint* li = list_i[sg];

    for (uint s = lane; s < k; s += SIMD) {
        ld[s] = INFINITY;
        li[s] = UINT_MAX;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const ulong q_offset = ulong(q) * 3;
    const float qx = query[q_offset];
    const float qy = query[q_offset + 1];
    const float qz = query[q_offset + 2];
    const bool query_finite = isfinite(qx) && isfinite(qy) && isfinite(qz);
    const uint chunks = n == 0 ? 0 : (n - 1) / SIMD + 1;
    if (query_finite && chunks != 0) {
        // A coprime stride visits every chunk once and avoids pathological
        // insertion rates when the input is stored in spatial order.
        uint step = uint(float(chunks) * 0.6180339887f + 0.5f) % chunks;
        if (step == 0) step = 1;
        while (gcd_uint(step, chunks) != 1) {
            step = (step + 1) % chunks;
            if (step == 0) step = 1;
        }
        uint chunk = 0;
        for (uint c = 0; c < chunks; ++c) {
            const uint j = chunk * SIMD + lane;
            const ulong next_chunk = ulong(chunk) + ulong(step);
            chunk = uint(next_chunk >= ulong(chunks) ? next_chunk - ulong(chunks) : next_chunk);
            float d = INFINITY;
            if (j < n) {
                const ulong x_offset = (x_begin + ulong(j)) * 3;
                const float dx = ref[x_offset] - qx;
                const float dy = ref[x_offset + 1] - qy;
                const float dz = ref[x_offset + 2] - qz;
                d = dx * dx;
                d = d + dy * dy;
                d = d + dz * dz;
            }
            const bool candidate = j < n && isfinite(d) && before(d, j, ld[k - 1], li[k - 1]);
            ulong pending = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(candidate)));
            while (pending != 0) {
                const ushort src = ushort(ctz(pending));
                pending &= pending - 1;
                const float cd = simd_shuffle(d, src);
                const uint ci = simd_shuffle(j, src);
                if (!before(cd, ci, ld[k - 1], li[k - 1])) continue;

                uint below = 0;
                for (uint s = lane; s < k; s += SIMD) {
                    below += before(ld[s], li[s], cd, ci) ? 1 : 0;
                }
                const uint pos = simd_sum(below);
                float keep_d[SLOTS];
                uint keep_i[SLOTS];
                for (uint t = 0; t < SLOTS; ++t) {
                    const uint s = lane + t * SIMD;
                    if (s > pos && s < k) {
                        keep_d[t] = ld[s - 1];
                        keep_i[t] = li[s - 1];
                    }
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
                for (uint t = 0; t < SLOTS; ++t) {
                    const uint s = lane + t * SIMD;
                    if (s > pos && s < k) {
                        ld[s] = keep_d[t];
                        li[s] = keep_i[t];
                    }
                }
                if (lane == 0) {
                    ld[pos] = cd;
                    li[pos] = ci;
                }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }

    const ulong row = ulong(q) * ulong(k);
    for (uint s = lane; s < k; s += SIMD) {
        out_idx[row + s] = li[s] == UINT_MAX ? -1 : long(x_begin + ulong(li[s]));
    }
}

// Match the dense Ball Query floating-point policy: explicit FMA accumulation
// and a normalized comparison for tiny radii. Metal may flush subnormals.
static inline float squared_norm(float3 delta) {
    float sum = delta.x * delta.x;
    sum = fma(delta.y, delta.y, sum);
    return fma(delta.z, delta.z, sum);
}

template <typename Scalar>
static inline float3 load_xyz(device const Scalar* points, ulong point) {
    const ulong offset = point * 3;
    return float3(float(points[offset]), float(points[offset + 1]), float(points[offset + 2]));
}

template <typename Scalar>
static inline void flat_radius_impl(
    device const Scalar* query,
    device const Scalar* ref,
    device const long* ptr_y,
    device const long* ptr_x,
    device long* out_idx,
    constant long& query_count,
    constant long& batch_count,
    constant long& K,
    constant float& radius_sq,
    constant float& radius,
    constant long& ignore_same_index,
    uint q,
    uint lane)
{
    if (ulong(q) >= ulong(query_count)) return;
    const uint b = batch_for_query(ptr_y, uint(batch_count), ulong(q));
    const ulong x_begin = ulong(ptr_x[b]);
    const ulong x_end = ulong(ptr_x[b + 1]);
    const ulong k = ulong(K);
    const ulong row = ulong(q) * k;
    uint found = 0;
    if (radius > 0.0f) {
        const float3 qp = load_xyz(query, ulong(q));
        if (all(isfinite(qp))) {
            for (ulong base = x_begin; base < x_end; base += SIMD) {
                const ulong j = base + ulong(lane);
                bool within = false;
                if (j < x_end && !(ignore_same_index && j == ulong(q))) {
                    const float3 xp = load_xyz(ref, j);
                    if (all(isfinite(xp))) {
                        const float3 delta = qp - xp;
                        if (all(isfinite(delta))) {
                            if (radius < 0x1p-50f || !isfinite(radius_sq)) {
                                within = squared_norm(delta / radius) < 1.0f;
                            } else {
                                within = squared_norm(delta) < radius_sq;
                            }
                        }
                    }
                }

                // All SIMD lanes participate. Inclusive chunk order plus the
                // exclusive prefix give the first K matching reference indices
                // in their original input order, even in a partially full block.
                const uint prefix = simd_prefix_exclusive_sum(uint(within));
                const uint block_count = simd_sum(uint(within));
                if (within && ulong(found) + ulong(prefix) < k) {
                    out_idx[row + ulong(found) + ulong(prefix)] = long(j);
                }
                found = uint(min(k, ulong(found) + ulong(block_count)));
                if (ulong(found) == k) break;
            }
        }
    }
    for (ulong s = ulong(found) + ulong(lane); s < k; s += SIMD) out_idx[row + s] = -1;
}

kernel void flat_radius_indices_f32(
    device const float* query [[buffer(0)]],
    device const float* ref [[buffer(1)]],
    device const long* ptr_y [[buffer(2)]],
    device const long* ptr_x [[buffer(3)]],
    device long* out_idx [[buffer(4)]],
    constant long& query_count [[buffer(5)]],
    constant long& batch_count [[buffer(6)]],
    constant long& K [[buffer(7)]],
    constant float& radius_sq [[buffer(8)]],
    constant float& radius [[buffer(9)]],
    constant long& ignore_same_index [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    #pragma clang fp contract(off)
    const ulong query_index = ulong(group.x) * QUERIES_PER_GROUP + ulong(sg);
    if (query_index >= ulong(query_count)) return;
    const uint q = uint(query_index);
    flat_radius_impl(query, ref, ptr_y, ptr_x, out_idx, query_count,
                     batch_count, K, radius_sq, radius, ignore_same_index, q, lane);
}

kernel void flat_radius_indices_f16(
    device const half* query [[buffer(0)]],
    device const half* ref [[buffer(1)]],
    device const long* ptr_y [[buffer(2)]],
    device const long* ptr_x [[buffer(3)]],
    device long* out_idx [[buffer(4)]],
    constant long& query_count [[buffer(5)]],
    constant long& batch_count [[buffer(6)]],
    constant long& K [[buffer(7)]],
    constant float& radius_sq [[buffer(8)]],
    constant float& radius [[buffer(9)]],
    constant long& ignore_same_index [[buffer(10)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    #pragma clang fp contract(off)
    const ulong query_index = ulong(group.x) * QUERIES_PER_GROUP + ulong(sg);
    if (query_index >= ulong(query_count)) return;
    const uint q = uint(query_index);
    flat_radius_impl(query, ref, ptr_y, ptr_x, out_idx, query_count,
                     batch_count, K, radius_sq, radius, ignore_same_index, q, lane);
}
