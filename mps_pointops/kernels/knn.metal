#include <metal_stdlib>
using namespace metal;

// Brute-force k nearest neighbors.
//
// One simdgroup per query. The 32 lanes scan the reference points 32 at a
// time. Each query keeps its current k nearest points as a list sorted by
// (squared distance, index) in threadgroup memory. A point is inserted only
// if it beats the current k-th entry, which after the first few chunks is
// rare, so most of the work is computing distances.
//
// Chunks of 32 points are visited in a scrambled order (chunk c, c + stride,
// c + 2 * stride, ... mod the chunk count, with stride coprime to it). Scans
// in storage order are slow on real scans, whose points are stored in spatial
// order: walking toward the query makes almost every point a new nearest one.
// The result does not depend on the order because the list is sorted by
// (distance, index).

constant constexpr uint SIMD = 32;
constant constexpr uint QUERIES_PER_GROUP = 8;  // simdgroups per threadgroup
constant constexpr uint MAX_K = 256;
constant constexpr uint SLOTS = MAX_K / SIMD;   // list entries per lane

// (d, i) < (e, j) in (distance, index) order.
static inline bool before(float d, uint i, float e, uint j) {
    return d < e || (d == e && i < j);
}

kernel void knn(
    device const float* query    [[buffer(0)]],  // (B, M, 3)
    device const float* ref      [[buffer(1)]],  // (B, N, 3)
    device float*       out_dist [[buffer(2)]],  // (B, M, k) Euclidean
    device long*        out_idx  [[buffer(3)]],  // (B, M, k)
    constant long&      M        [[buffer(4)]],
    constant long&      N        [[buffer(5)]],
    constant long&      K        [[buffer(6)]],
    constant long&      stride   [[buffer(7)]],  // coprime to the chunk count
    uint2 group [[threadgroup_position_in_grid]],
    uint  lane  [[thread_index_in_simdgroup]],
    uint  sg    [[simdgroup_index_in_threadgroup]])
{
    // Same rounding as the reference: ((dx^2 + dy^2) + dz^2), no FMA.
    #pragma clang fp contract(off)

    threadgroup float list_d[QUERIES_PER_GROUP][MAX_K];
    threadgroup uint  list_i[QUERIES_PER_GROUP][MAX_K];

    const uint b = group.y;
    const uint q = group.x * QUERIES_PER_GROUP + sg;
    if (q >= uint(M)) {
        return;  // only simdgroup barriers below, so idle simdgroups can leave
    }
    const uint n = uint(N);
    const uint k = uint(K);
    threadgroup float* ld = list_d[sg];
    threadgroup uint*  li = list_i[sg];

    device const float* qp = query + (ulong(b) * ulong(M) + q) * 3;
    device const float* p = ref + ulong(b) * ulong(N) * 3;
    const float qx = qp[0], qy = qp[1], qz = qp[2];

    for (uint s = lane; s < k; s += SIMD) {
        ld[s] = INFINITY;
        li[s] = UINT_MAX;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint chunks = (n + SIMD - 1) / SIMD;
    const uint step = uint(stride);
    uint chunk = 0;
    for (uint c = 0; c < chunks; ++c) {
        const uint j = chunk * SIMD + lane;
        chunk += step;
        if (chunk >= chunks) {
            chunk -= chunks;
        }
        float d = INFINITY;
        if (j < n) {
            const float dx = p[j * 3 + 0] - qx;
            const float dy = p[j * 3 + 1] - qy;
            const float dz = p[j * 3 + 2] - qz;
            d = dx * dx;
            d = d + dy * dy;
            d = d + dz * dz;
        }
        const bool candidate = j < n && before(d, j, ld[k - 1], li[k - 1]);
        ulong pending = static_cast<ulong>(static_cast<simd_vote::vote_t>(simd_ballot(candidate)));

        // Insert candidates one at a time, in lane order. Every lane runs the
        // same iterations because pending is uniform across the simdgroup.
        while (pending != 0) {
            const ushort src = ushort(ctz(pending));
            pending &= pending - 1;
            const float cd = simd_shuffle(d, src);
            const uint ci = simd_shuffle(j, src);
            if (!before(cd, ci, ld[k - 1], li[k - 1])) {
                continue;  // an earlier insert in this chunk raised the bar
            }

            uint below = 0;
            for (uint s = lane; s < k; s += SIMD) {
                below += before(ld[s], li[s], cd, ci) ? 1 : 0;
            }
            const uint pos = simd_sum(below);

            // Shift entries [pos, k - 1) one slot right, then write the new one.
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

    const ulong row = (ulong(b) * ulong(M) + q) * ulong(k);
    for (uint s = lane; s < k; s += SIMD) {
        out_dist[row + s] = sqrt(ld[s]);
        out_idx[row + s] = long(li[s]);
    }
}
