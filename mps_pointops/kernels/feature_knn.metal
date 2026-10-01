#include <metal_stdlib>
using namespace metal;

// Feature-space squared L2 search. One SIMD group owns one query. Distance is
// accumulated directly in increasing dimension order, with contraction off:
// s = fl32(...fl32(fl32(delta[0]^2) + fl32(delta[1]^2))...). This avoids the
// cancellation of the matrix-multiply identity ||x||^2+||y||^2-2<x,y>.
// A finite candidate's key is (squared distance, reference index). Sorting by
// that key makes the result independent of the scrambled reference visit order.
// An invalid/non-finite candidate leaves its output slot as (inf, -1).

constant constexpr uint SIMD = 32;
constant constexpr uint QUERIES_PER_GROUP = 8;
constant constexpr uint MAX_K = 256;
constant constexpr uint SLOTS = MAX_K / SIMD;

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

static inline uint visit_step(uint chunks) {
    uint step = uint(float(chunks) * 0.6180339887f + 0.5f) % chunks;
    if (step == 0) step = 1;
    while (gcd_uint(step, chunks) != 1) {
        step = (step + 1) % chunks;
        if (step == 0) step = 1;
    }
    return step;
}

static inline uint batch_for_query(device const long* ptr_y, uint batch_count, ulong query) {
    uint lo = 0;
    uint hi = batch_count;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        if (ulong(ptr_y[mid + 1]) <= query) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline void feature_scan(
    device const float* query,
    device const float* ref,
    uint n,
    uint dim,
    uint k,
    uint lane,
    threadgroup float* ld,
    threadgroup uint* li)
{
    #pragma clang fp contract(off)
    for (uint slot = lane; slot < k; slot += SIMD) {
        ld[slot] = INFINITY;
        li[slot] = UINT_MAX;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    bool query_finite = true;
    for (uint t = 0; t < dim; ++t) {
        query_finite = query_finite && isfinite(query[t]);
    }
    const uint chunks = n == 0 ? 0 : (n - 1) / SIMD + 1;
    if (!query_finite || chunks == 0) return;

    const uint step = visit_step(chunks);
    uint chunk = 0;
    for (uint c = 0; c < chunks; ++c) {
        const uint j = chunk * SIMD + lane;
        const ulong next_chunk = ulong(chunk) + ulong(step);
        chunk = uint(next_chunk >= ulong(chunks) ? next_chunk - ulong(chunks) : next_chunk);
        float d = INFINITY;
        if (j < n) {
            device const float* point = ref + ulong(j) * ulong(dim);
            const float first = point[0] - query[0];
            d = first * first;
            for (uint t = 1; t < dim; ++t) {
                const float delta = point[t] - query[t];
                const float term = delta * delta;
                d = d + term;
            }
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
            for (uint slot = lane; slot < k; slot += SIMD) {
                below += before(ld[slot], li[slot], cd, ci) ? 1 : 0;
            }
            const uint pos = simd_sum(below);

            float keep_d[SLOTS];
            uint keep_i[SLOTS];
            for (uint t = 0; t < SLOTS; ++t) {
                const uint slot = lane + t * SIMD;
                if (slot > pos && slot < k) {
                    keep_d[t] = ld[slot - 1];
                    keep_i[t] = li[slot - 1];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint t = 0; t < SLOTS; ++t) {
                const uint slot = lane + t * SIMD;
                if (slot > pos && slot < k) {
                    ld[slot] = keep_d[t];
                    li[slot] = keep_i[t];
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

kernel void feature_knn_dense(
    device const float* query [[buffer(0)]],
    device const float* ref [[buffer(1)]],
    device float* out_dist [[buffer(2)]],
    device long* out_idx [[buffer(3)]],
    constant long& M [[buffer(4)]],
    constant long& N [[buffer(5)]],
    constant long& K [[buffer(6)]],
    constant long& D [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float list_d[QUERIES_PER_GROUP][MAX_K];
    threadgroup uint list_i[QUERIES_PER_GROUP][MAX_K];
    const uint q = group.x * QUERIES_PER_GROUP + sg;
    if (q >= uint(M)) return;

    const uint b = group.y;
    const uint dim = uint(D);
    device const float* qp = query + (ulong(b) * ulong(M) + q) * ulong(dim);
    device const float* rp = ref + ulong(b) * ulong(N) * ulong(dim);
    threadgroup float* ld = list_d[sg];
    threadgroup uint* li = list_i[sg];
    feature_scan(qp, rp, uint(N), dim, uint(K), lane, ld, li);

    const ulong row = (ulong(b) * ulong(M) + q) * ulong(K);
    for (uint slot = lane; slot < uint(K); slot += SIMD) {
        out_dist[row + slot] = sqrt(ld[slot]);
        out_idx[row + slot] = li[slot] == UINT_MAX ? -1 : long(li[slot]);
    }
}

kernel void feature_knn_flat(
    device const float* query [[buffer(0)]],
    device const float* ref [[buffer(1)]],
    device const long* ptr_y [[buffer(2)]],
    device const long* ptr_x [[buffer(3)]],
    device long* out_idx [[buffer(4)]],
    constant long& query_count [[buffer(5)]],
    constant long& batch_count [[buffer(6)]],
    constant long& K [[buffer(7)]],
    constant long& D [[buffer(8)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    threadgroup float list_d[QUERIES_PER_GROUP][MAX_K];
    threadgroup uint list_i[QUERIES_PER_GROUP][MAX_K];
    const ulong query_index = ulong(group.x) * QUERIES_PER_GROUP + ulong(sg);
    if (query_index >= ulong(query_count)) return;

    const uint dim = uint(D);
    const uint b = batch_for_query(ptr_y, uint(batch_count), query_index);
    const ulong x_begin = ulong(ptr_x[b]);
    const uint n = uint(ulong(ptr_x[b + 1]) - x_begin);
    device const float* qp = query + query_index * ulong(dim);
    device const float* rp = ref + x_begin * ulong(dim);
    threadgroup float* ld = list_d[sg];
    threadgroup uint* li = list_i[sg];
    feature_scan(qp, rp, n, dim, uint(K), lane, ld, li);

    const ulong row = query_index * ulong(K);
    for (uint slot = lane; slot < uint(K); slot += SIMD) {
        out_idx[row + slot] = li[slot] == UINT_MAX ? -1 : long(x_begin + ulong(li[slot]));
    }
}
