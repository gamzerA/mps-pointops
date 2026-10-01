#include <metal_stdlib>
using namespace metal;

// One SIMD group owns one x row. The legacy CUDA kernel uses 1024 lanes and
// scans each lane's stride-1024 subsequence, then preserves the lower lane on
// equal distance. Its tie key is therefore (distance, local index % 1024,
// local index), which is not always the first global index for >1024 points.
constant constexpr uint SIMD = 32;
constant constexpr uint QUERIES_PER_GROUP = 8;
constant constexpr uint CUDA_LANES = 1024;
constant constexpr float INITIAL_DISTANCE = 1e38f;

static inline uint batch_for_query(device const long* ptr, uint batches, ulong row) {
    uint lo = 0;
    uint hi = batches;
    while (lo < hi) {
        const uint mid = lo + (hi - lo) / 2;
        if (ulong(ptr[mid + 1]) <= row) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

kernel void nearest_flat_f32(
    device const float* x [[buffer(0)]],
    device const float* y [[buffer(1)]],
    device const long* ptr_x [[buffer(2)]],
    device const long* ptr_y [[buffer(3)]],
    device long* out [[buffer(4)]],
    constant long& query_count [[buffer(5)]],
    constant long& batch_count [[buffer(6)]],
    constant long& D [[buffer(7)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]])
{
    #pragma clang fp contract(off)
    const ulong row = ulong(group.x) * QUERIES_PER_GROUP + ulong(sg);
    if (row >= ulong(query_count)) return;
    const uint batch = batch_for_query(ptr_x, uint(batch_count), row);
    const ulong begin = ulong(ptr_y[batch]);
    const ulong count = ulong(ptr_y[batch + 1]) - begin;
    const uint dim = uint(D);
    device const float* query = x + row * ulong(dim);

    float best = INITIAL_DISTANCE;
    uint index = 0;
    uint cuda_lane = UINT_MAX;
    for (ulong local = ulong(lane); local < count; local += SIMD) {
        device const float* point = y + (begin + local) * ulong(dim);
        float distance = 0.0f;
        for (uint d = 0; d < dim; ++d) {
            const float delta = query[d] - point[d];
            const float term = delta * delta;
            distance = distance + term;
        }
        const uint candidate_lane = uint(local % CUDA_LANES);
        if (distance < best ||
            (distance == best && distance < INITIAL_DISTANCE &&
             (candidate_lane < cuda_lane ||
              (candidate_lane == cuda_lane && local < ulong(index))))) {
            best = distance;
            index = uint(local);
            cuda_lane = candidate_lane;
        }
    }
    const float minimum = simd_min(best);
    const uint winning_lane = simd_min(best == minimum ? cuda_lane : UINT_MAX);
    const uint winner = simd_min(best == minimum && cuda_lane == winning_lane ? index : UINT_MAX);
    if (lane == 0) {
        // No candidate beats the CUDA kernel's 1e38 initialization. Its
        // reduction leaves the default global index 0, even for later batches.
        out[row] = minimum < INITIAL_DISTANCE ? long(begin + ulong(winner)) : 0;
    }
}
