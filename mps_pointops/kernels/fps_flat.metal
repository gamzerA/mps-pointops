#include <metal_stdlib>
using namespace metal;

// Flat variable-length FPS. One threadgroup owns one cloud and runs all of
// its samples, preserving the reduction order and tie policy of fps.metal.
// Empty clouds with zero requested samples exit uniformly before any barrier.
kernel void fps_flat(
    device const float* xyz      [[buffer(0)]],  // (Total_N, 3)
    device const long* ptr       [[buffer(1)]],  // (B + 1,)
    device const long* out_ptr   [[buffer(2)]],  // (B + 1,)
    device const long* starts    [[buffer(3)]],  // (B,) local indices
    device float*       min_d2   [[buffer(4)]],  // (Total_N,) scratch
    device long*        out      [[buffer(5)]],  // (out_ptr[B],) global indices
    uint2 group     [[threadgroup_position_in_grid]],
    uint2 group_dim [[threads_per_threadgroup]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]],
    uint  sg        [[simdgroup_index_in_threadgroup]],
    uint  n_sg      [[simdgroups_per_threadgroup]])
{
    // Same rounding as fps.metal: ((dx^2 + dy^2) + dz^2), without FMA.
    #pragma clang fp contract(off)

    const uint b = group.y;
    const long input_start = ptr[b];
    const long input_count = ptr[b + 1] - input_start;
    const long output_start = out_ptr[b];
    const long output_count = out_ptr[b + 1] - output_start;
    if (output_count == 0) {
        return;
    }

    const uint n = uint(input_count);  // host validates n <= UINT_MAX
    const uint T = group_dim.x;
    device const float* p = xyz + ulong(input_start) * 3;
    device float* md = min_d2 + ulong(input_start);
    device long* o = out + ulong(output_start);

    threadgroup float sg_best[32];
    threadgroup uint sg_index[32];
    threadgroup uint farthest_shared;

    uint farthest = uint(starts[b]);
    for (long step = 0; step < output_count; ++step) {
        if (tid == 0) {
            o[step] = input_start + long(farthest);
        }
        const ulong centroid_offset = ulong(farthest) * 3;
        const float cx = p[centroid_offset + 0];
        const float cy = p[centroid_offset + 1];
        const float cz = p[centroid_offset + 2];

        float best = -INFINITY;
        uint best_index = UINT_MAX;
        for (ulong j = tid; j < ulong(n); j += T) {
            const ulong offset = j * 3;
            const float x = p[offset + 0];
            const float y = p[offset + 1];
            const float z = p[offset + 2];
            const float dx = x - cx;
            const float dy = y - cy;
            const float dz = z - cz;
            float d = dx * dx;
            d = d + dy * dy;
            d = d + dz * dz;
            const float m = step == 0 ? d : min(md[j], d);
            md[j] = m;
            // Local j increases within each thread; strict > keeps low index.
            if (m > best) {
                best = m;
                best_index = uint(j);
            }
        }

        float v = simd_max(best);
        uint i = simd_min(best == v ? best_index : UINT_MAX);
        if (lane == 0) {
            sg_best[sg] = v;
            sg_index[sg] = i;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg == 0) {
            v = lane < n_sg ? sg_best[lane] : -INFINITY;
            i = lane < n_sg ? sg_index[lane] : UINT_MAX;
            const float w = simd_max(v);
            i = simd_min(v == w ? i : UINT_MAX);
            if (lane == 0) {
                farthest_shared = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Once all distances tie at zero, continue with the first local point.
        farthest = farthest_shared == UINT_MAX ? 0 : farthest_shared;
    }
}
