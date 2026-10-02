// SPDX-License-Identifier: Apache-2.0
// Experimental sorted-Morton radius query. One thread owns one query row.
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

static inline ulong spread_21(uint value) {
    ulong bits = ulong(value & 0x1fffffu);
    bits = (bits | (bits << 32)) & 0x1f00000000fffful;
    bits = (bits | (bits << 16)) & 0x1f0000ff0000fful;
    bits = (bits | (bits << 8)) & 0x100f00f00f00f00ful;
    bits = (bits | (bits << 4)) & 0x10c30c30c30c30c3ul;
    return (bits | (bits << 2)) & 0x1249249249249249ul;
}

static inline long key_3d(uint x, uint y, uint z) {
    return long(spread_21(x) | (spread_21(y) << 1) | (spread_21(z) << 2));
}

static inline ulong lower_bound_key(device const long *keys, ulong count, long target) {
    ulong lo = 0;
    ulong hi = count;
    while (lo < hi) {
        const ulong mid = lo + (hi - lo) / 2;
        if (keys[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline ulong upper_bound_key(device const long *keys, ulong count, long target) {
    ulong lo = 0;
    ulong hi = count;
    while (lo < hi) {
        const ulong mid = lo + (hi - lo) / 2;
        if (keys[mid] <= target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static inline float squared_norm(float3 delta) {
    float sum = delta.x * delta.x;
    sum = fma(delta.y, delta.y, sum);
    return fma(delta.z, delta.z, sum);
}

kernel void spatial_radius_first_k_f32(
    device const float *query [[buffer(0)]],
    device const float *ref [[buffer(1)]],
    device const long *sorted_keys [[buffer(2)]],
    device const long *sorted_indices [[buffer(3)]],
    device const float *origin [[buffer(4)]],
    device long *out_indices [[buffer(5)]],
    device uchar *status [[buffer(6)]],
    constant long &query_count [[buffer(7)]],
    constant long &ref_count [[buffer(8)]],
    constant long &limit [[buffer(9)]],
    constant float &radius [[buffer(10)]],
    constant float &radius_sq [[buffer(11)]],
    constant float &cell_size [[buffer(12)]],
    uint q [[thread_position_in_grid]]) {
    if (ulong(q) >= ulong(query_count)) return;
    const ulong row = ulong(q) * ulong(limit);
    for (long s = 0; s < limit; ++s) out_indices[row + ulong(s)] = -1;
    status[q] = uchar(0);
    if (limit == 0 || radius <= 0.0f || ref_count == 0) return;

    const ulong qoffset = ulong(q) * 3;
    const float3 qp = float3(query[qoffset], query[qoffset + 1], query[qoffset + 2]);
    const float3 base = float3(origin[0], origin[1], origin[2]);
    if (!all(isfinite(qp))) return;

    const float3 low_cell = floor((qp - radius - base) / cell_size);
    const float3 high_cell = floor((qp + radius - base) / cell_size);
    if (!all(isfinite(low_cell)) || !all(isfinite(high_cell)) ||
        any(low_cell < -2.0f) || any(high_cell >= 2097153.0f)) {
        status[q] = uchar(1);  // explicit fallback required
        return;
    }
    const int3 lo = max(int3(low_cell) - 1, int3(0));
    const int3 hi = min(int3(high_cell) + 1, int3(2097151));
    if (any(hi < lo)) return;
    const long3 span = long3(hi - lo + 1);
    if (span.x > 4096 || span.y > 4096 || span.z > 4096 ||
        span.x * span.y > 4096 / span.z) {
        status[q] = uchar(1);  // do not hide a pathological cell traversal
        return;
    }

    uint found = 0;
    for (int cx = lo.x; cx <= hi.x; ++cx) {
        for (int cy = lo.y; cy <= hi.y; ++cy) {
            for (int cz = lo.z; cz <= hi.z; ++cz) {
                const long key = key_3d(uint(cx), uint(cy), uint(cz));
                const ulong first = lower_bound_key(sorted_keys, ulong(ref_count), key);
                const ulong past = upper_bound_key(sorted_keys, ulong(ref_count), key);
                for (ulong position = first; position < past; ++position) {
                    const ulong source = ulong(sorted_indices[position]);
                    const ulong offset = source * 3;
                    const float3 xp = float3(ref[offset], ref[offset + 1], ref[offset + 2]);
                    const float3 delta = qp - xp;
                    bool within;
                    if (radius < 0x1p-50f || !isfinite(radius_sq)) {
                        within = squared_norm(delta / radius) < 1.0f;
                    } else {
                        within = squared_norm(delta) < radius_sq;
                    }
                    if (!within) continue;
                    // Cell order is spatial, not source order. Keep the K
                    // smallest original indices, sorted, to preserve the
                    // torch_cluster first-K collection contract exactly.
                    uint insert = 0;
                    while (insert < found && ulong(out_indices[row + insert]) < source) ++insert;
                    if (insert >= ulong(limit)) continue;
                    const uint new_found = min(uint(limit), found + 1);
                    for (uint s = new_found - 1; s > insert; --s) {
                        out_indices[row + s] = out_indices[row + s - 1];
                    }
                    out_indices[row + insert] = long(source);
                    found = new_found;
                }
            }
        }
    }
}
