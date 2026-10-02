#include <metal_stdlib>
using namespace metal;

// CPU-generated output CSR: each output scalar has one writer, so neither
// float atomics nor an inter-thread reduction affect accumulation order.
// Weight layout is PyTorch Conv3d [Cout, Cin, K0, K1, K2].
kernel void subm_conv3d_f32(
    device const float* features [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const long* ptr [[buffer(2)]],
    device const long* sources [[buffer(3)]],
    device const long* offsets [[buffer(4)]],
    device const float* bias [[buffer(5)]],
    device float* output [[buffer(6)]],
    constant long& row_count [[buffer(7)]],
    constant long& in_channels [[buffer(8)]],
    constant long& out_channels [[buffer(9)]],
    constant long& kernel_volume [[buffer(10)]],
    constant int& has_bias [[buffer(11)]],
    uint tid [[thread_position_in_grid]])
{
    #pragma METAL fp math_mode(safe)
    const ulong scalar = ulong(tid);
    if (scalar >= ulong(row_count) * ulong(out_channels)) return;
    const ulong row = scalar / ulong(out_channels);
    const ulong out_channel = scalar % ulong(out_channels);
    float total = has_bias ? bias[out_channel] : 0.0f;
    for (long pair = ptr[row]; pair < ptr[row + 1]; ++pair) {
        const ulong input_row = ulong(sources[pair]);
        const ulong offset = ulong(offsets[pair]);
        for (ulong channel = 0; channel < ulong(in_channels); ++channel) {
            const float value = features[input_row * ulong(in_channels) + channel];
            const ulong weight_index =
                (out_channel * ulong(in_channels) + channel) * ulong(kernel_volume) + offset;
            total = fma(value, weights[weight_index], total);
        }
    }
    output[scalar] = total;
}
