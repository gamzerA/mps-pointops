#include <metal_stdlib>
using namespace metal;

// Original Metal implementation of PointNet++ feature propagation primitives.
// All offsets are computed from the validated dense shapes. The Python entry
// points require float32 coordinates, features, and weights, and int32 indices.

kernel void three_nn_f32(
    device const float* unknown [[buffer(0)]],
    device const float* known [[buffer(1)]],
    device float* distances [[buffer(2)]],
    device int* indices [[buffer(3)]],
    constant long& N [[buffer(4)]],
    constant long& M [[buffer(5)]],
    uint tid [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    const ulong b = ulong(tid) / ulong(N);
    const ulong q = ulong(tid) % ulong(N);
    const ulong uoff = (b * ulong(N) + q) * 3;
    const ulong kbase = b * ulong(M) * 3;
    const float ux = unknown[uoff], uy = unknown[uoff + 1], uz = unknown[uoff + 2];
    float d0 = INFINITY, d1 = INFINITY, d2 = INFINITY;
    int i0 = -1, i1 = -1, i2 = -1;
    for (long j = 0; j < M; ++j) {
        const ulong off = kbase + ulong(j) * 3;
        const float dx = ux - known[off];
        const float dy = uy - known[off + 1];
        const float dz = uz - known[off + 2];
        float d = dx * dx;
        d = d + dy * dy;
        d = d + dz * dz;
        // Strict comparisons retain the first input index for equal distances.
        if (i0 < 0 || d < d0) {
            d2 = d1; i2 = i1;
            d1 = d0; i1 = i0;
            d0 = d; i0 = int(j);
        } else if (i1 < 0 || d < d1) {
            d2 = d1; i2 = i1;
            d1 = d; i1 = int(j);
        } else if (i2 < 0 || d < d2) {
            d2 = d; i2 = int(j);
        }
    }
    distances[uoff] = sqrt(d0);
    distances[uoff + 1] = sqrt(d1);
    distances[uoff + 2] = sqrt(d2);
    indices[uoff] = i0;
    indices[uoff + 1] = i1;
    indices[uoff + 2] = i2;
}

kernel void three_interpolate_f32(
    device const float* features [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant long& C [[buffer(4)]],
    constant long& M [[buffer(5)]],
    constant long& N [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    const ulong n = ulong(tid) % ulong(N);
    const ulong c = (ulong(tid) / ulong(N)) % ulong(C);
    const ulong b = ulong(tid) / (ulong(N) * ulong(C));
    const ulong triplet = (b * ulong(N) + n) * 3;
    const ulong base = (b * ulong(C) + c) * ulong(M);
    const float a = features[base + ulong(indices[triplet])] * weights[triplet];
    const float b1 = features[base + ulong(indices[triplet + 1])] * weights[triplet + 1];
    const float c1 = features[base + ulong(indices[triplet + 2])] * weights[triplet + 2];
    output[tid] = (a + b1) + c1;
}

// Float atomic addition via integer CAS. This handles repeated source indices
// without dropping contributions. As with CUDA atomicAdd, summation order can
// vary, so low bits of the gradient are not deterministic.
static inline void add_float(device atomic_uint* slot, float value) {
    uint old_bits = atomic_load_explicit(slot, memory_order_relaxed);
    uint new_bits;
    do {
        new_bits = as_type<uint>(as_type<float>(old_bits) + value);
    } while (!atomic_compare_exchange_weak_explicit(
        slot, &old_bits, new_bits, memory_order_relaxed, memory_order_relaxed));
}

kernel void three_interpolate_backward_f32(
    device const float* grad_output [[buffer(0)]],
    device const int* indices [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device atomic_uint* grad_features [[buffer(3)]],
    constant long& C [[buffer(4)]],
    constant long& M [[buffer(5)]],
    constant long& N [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong n = ulong(tid) % ulong(N);
    const ulong c = (ulong(tid) / ulong(N)) % ulong(C);
    const ulong b = ulong(tid) / (ulong(N) * ulong(C));
    const ulong triplet = (b * ulong(N) + n) * 3;
    const ulong base = (b * ulong(C) + c) * ulong(M);
    const float g = grad_output[tid];
    add_float(grad_features + base + ulong(indices[triplet]), g * weights[triplet]);
    add_float(grad_features + base + ulong(indices[triplet + 1]), g * weights[triplet + 1]);
    add_float(grad_features + base + ulong(indices[triplet + 2]), g * weights[triplet + 2]);
}
