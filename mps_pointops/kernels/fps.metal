#include <metal_stdlib>
using namespace metal;

// Farthest point sampling.
//
// One threadgroup per batch element runs all npoint steps, so the whole
// sampling is a single dispatch. Thread t owns points t, t + T, t + 2T, ...
// and keeps their running minimum squared distance to the sampled set in
// min_d2. Each step reduces (distance, index) to the farthest point with
// simdgroup and threadgroup reductions. Ties go to the smaller index, like
// torch.argmax.
//
// With skip_near_origin set, points with x^2 + y^2 + z^2 <= 1e-3 are never
// picked (only the start index can be one), matching pointnet2_ops. The float
// comparison mag < 1e-3f is the same as pointnet2_ops' mag <= 1e-3 in double:
// fl32(1e-3) is just above 1e-3, and no float lies between them.
//
// The reductions assume 32-wide simdgroups, which ops.py checks at load time.
kernel void furthest_point_sample(
    device const float* xyz    [[buffer(0)]],  // (B, N, 3)
    device float*       min_d2 [[buffer(1)]],  // (B, N) scratch
    device long*        out    [[buffer(2)]],  // (B, npoint)
    constant long&      N      [[buffer(3)]],
    constant long&      npoint [[buffer(4)]],
    constant long&      start  [[buffer(5)]],
    constant long&      skip_near_origin [[buffer(6)]],
    uint2 group     [[threadgroup_position_in_grid]],
    uint2 group_dim [[threads_per_threadgroup]],
    uint  tid       [[thread_index_in_threadgroup]],
    uint  lane      [[thread_index_in_simdgroup]],
    uint  sg        [[simdgroup_index_in_threadgroup]],
    uint  n_sg      [[simdgroups_per_threadgroup]])
{
    // Same rounding as the reference: ((dx^2 + dy^2) + dz^2), no FMA.
    #pragma clang fp contract(off)

    const uint b = group.y;
    const uint T = group_dim.x;
    const uint n = uint(N);
    device const float* p = xyz + ulong(b) * ulong(N) * 3;
    device float* md = min_d2 + ulong(b) * ulong(N);
    device long* o = out + ulong(b) * ulong(npoint);

    threadgroup float sg_best[32];
    threadgroup uint  sg_index[32];
    threadgroup uint  farthest_shared;

    uint farthest = uint(start);
    for (long step = 0; step < npoint; ++step) {
        if (tid == 0) {
            o[step] = long(farthest);
        }
        const float cx = p[farthest * 3 + 0];
        const float cy = p[farthest * 3 + 1];
        const float cz = p[farthest * 3 + 2];

        float best = -INFINITY;
        uint best_index = UINT_MAX;
        for (uint j = tid; j < n; j += T) {
            const float x = p[j * 3 + 0];
            const float y = p[j * 3 + 1];
            const float z = p[j * 3 + 2];
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
            const float m = step == 0 ? d : min(md[j], d);
            md[j] = m;
            // j grows within a thread, so strict > keeps the smaller index.
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
                farthest_shared = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // UINT_MAX means every point was skipped; fall back to 0 like pointnet2_ops.
        farthest = farthest_shared == UINT_MAX ? 0 : farthest_shared;
    }
}
