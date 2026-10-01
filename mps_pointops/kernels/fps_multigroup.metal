#include <metal_stdlib>
using namespace metal;

// Large-cloud FPS path for B=1. Each group owns one contiguous chunk of
// points, updates running minimum distances, and emits one (value, index).
// A second dispatch reduces the group results to select the next centroid.
// The two dispatches are enqueued in program order for each sample. This
// cross-dispatch ordering is necessary: an in-kernel barrier would not
// synchronize distinct threadgroups.
kernel void fps_update_partials(
    device const float* xyz [[buffer(0)]],
    device float* min_d2 [[buffer(1)]],
    device const long* out [[buffer(2)]],
    device float* partial_d2 [[buffer(3)]],
    device uint* partial_idx [[buffer(4)]],
    constant long& N [[buffer(5)]],
    constant long& step [[buffer(6)]],
    constant long& chunk [[buffer(7)]],
    constant long& skip_near_origin [[buffer(8)]],
    uint gid [[threadgroup_position_in_grid]],
    uint T [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]])
{
    #pragma clang fp contract(off)
    threadgroup float sg_best[32];
    threadgroup uint sg_index[32];

    // N < 2^32 is checked by the host. Using an offset avoids overflow from
    // first + chunk for a final chunk near UINT_MAX.
    const uint first = gid * uint(chunk);
    const uint length = min(uint(chunk), uint(N) - first);
    const uint farthest = uint(out[step]);
    const ulong center_offset = ulong(farthest) * 3;
    const float cx = xyz[center_offset + 0];
    const float cy = xyz[center_offset + 1];
    const float cz = xyz[center_offset + 2];

    float best = -INFINITY;
    uint best_index = UINT_MAX;
    for (uint offset = tid; offset < length; offset += T) {
        const uint j = first + offset;
        const ulong point_offset = ulong(j) * 3;
        const float x = xyz[point_offset + 0];
        const float y = xyz[point_offset + 1];
        const float z = xyz[point_offset + 2];
        if (skip_near_origin != 0) {
            float mag = x * x;
            mag = mag + y * y;
            mag = mag + z * z;
            if (mag < 1e-3f) {
                continue;
            }
        }
        const float dx = x - cx;
        const float dy = y - cy;
        const float dz = z - cz;
        float d = dx * dx;
        d = d + dy * dy;
        d = d + dz * dz;
        const float m = step == 0 ? d : min(min_d2[j], d);
        min_d2[j] = m;
        if (m > best) {
            best = m;
            best_index = j;
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
            partial_d2[gid] = w;
            partial_idx[gid] = i;
        }
    }
}

// Reduction across partial maxima. Strict value max followed by minimum index
// has the same deterministic tie order as torch.argmax and the current kernel.
kernel void fps_reduce_partials(
    device const float* partial_d2 [[buffer(0)]],
    device const uint* partial_idx [[buffer(1)]],
    device long* out [[buffer(2)]],
    constant long& parts [[buffer(3)]],
    constant long& step [[buffer(4)]],
    uint T [[threads_per_threadgroup]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint n_sg [[simdgroups_per_threadgroup]])
{
    threadgroup float sg_best[32];
    threadgroup uint sg_index[32];

    float best = -INFINITY;
    uint best_index = UINT_MAX;
    for (uint p = tid; p < uint(parts); p += T) {
        const float v = partial_d2[p];
        const uint i = partial_idx[p];
        if (v > best || (v == best && i < best_index)) {
            best = v;
            best_index = i;
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
            out[step + 1] = i == UINT_MAX ? 0 : long(i);
        }
    }
}
