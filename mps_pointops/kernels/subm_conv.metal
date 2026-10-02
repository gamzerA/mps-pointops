#include <metal_stdlib>
using namespace metal;

// Output CSR may be constructed on CPU or MPS. Each output scalar has one
// writer, so neither float atomics nor inter-thread reduction affect its sum.
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

// A SubM rulebook with an odd centered kernel is symmetric: for every
// (offset k, source i, output o), its reverse pair has output i, source o,
// and offset (kernel_volume - 1 - k). Traverse that reverse pair in the
// output CSR so one thread owns each input-gradient scalar without atomics.
kernel void subm_conv3d_grad_features_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const long* ptr [[buffer(2)]],
    device const long* sources [[buffer(3)]],
    device const long* offsets [[buffer(4)]],
    device float* grad_features [[buffer(5)]],
    constant long& row_count [[buffer(6)]],
    constant long& in_channels [[buffer(7)]],
    constant long& out_channels [[buffer(8)]],
    constant long& kernel_volume [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    #pragma METAL fp math_mode(safe)
    const ulong scalar = ulong(tid);
    if (scalar >= ulong(row_count) * ulong(in_channels)) return;
    const ulong row = scalar / ulong(in_channels);
    const ulong in_channel = scalar % ulong(in_channels);
    float total = 0.0f;
    for (long pair = ptr[row]; pair < ptr[row + 1]; ++pair) {
        const ulong output_row = ulong(sources[pair]);
        const ulong reverse_offset = ulong(kernel_volume - 1 - offsets[pair]);
        for (ulong out_channel = 0; out_channel < ulong(out_channels); ++out_channel) {
            const ulong weight_index =
                (out_channel * ulong(in_channels) + in_channel) * ulong(kernel_volume)
                + reverse_offset;
            total = fma(
                grad_output[output_row * ulong(out_channels) + out_channel],
                weights[weight_index], total
            );
        }
    }
    grad_features[scalar] = total;
}

// The GPU rulebook provides offset-major pairs and a device-resident offset
// CSR. One thread owns each weight-gradient scalar; pair_count never needs a
// host readback and no floating-point atomic operation is required.
kernel void subm_conv3d_grad_weights_f32(
    device const float* features [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device const long* pairs [[buffer(2)]],
    device const long* offset_ptr [[buffer(3)]],
    device float* grad_weights [[buffer(4)]],
    constant long& in_channels [[buffer(5)]],
    constant long& out_channels [[buffer(6)]],
    constant long& kernel_volume [[buffer(7)]],
    uint tid [[thread_position_in_grid]])
{
    #pragma METAL fp math_mode(safe)
    const ulong scalar = ulong(tid);
    const ulong work = ulong(in_channels) * ulong(out_channels) * ulong(kernel_volume);
    if (scalar >= work) return;
    const ulong offset = scalar % ulong(kernel_volume);
    const ulong in_channel = (scalar / ulong(kernel_volume)) % ulong(in_channels);
    const ulong out_channel = scalar / (ulong(kernel_volume) * ulong(in_channels));
    float total = 0.0f;
    for (long pair = offset_ptr[offset]; pair < offset_ptr[offset + 1]; ++pair) {
        const ulong source_row = ulong(pairs[3 * pair + 1]);
        const ulong output_row = ulong(pairs[3 * pair + 2]);
        total = fma(
            grad_output[output_row * ulong(out_channels) + out_channel],
            features[source_row * ulong(in_channels) + in_channel], total
        );
    }
    grad_weights[scalar] = total;
}
