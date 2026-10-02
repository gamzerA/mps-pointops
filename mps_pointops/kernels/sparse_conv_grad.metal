#include <metal_stdlib>
using namespace metal;

// Ordinary stride has no SubM reverse-offset symmetry. CSR explicitly maps
// each input row to the forward output row and its original kernel offset.
kernel void sparse_conv3d_grad_features_f32(
    device const float* grad_output [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const long* input_ptr [[buffer(2)]],
    device const long* input_outputs [[buffer(3)]],
    device const long* input_offsets [[buffer(4)]],
    device float* grad_features [[buffer(5)]],
    constant long& input_count [[buffer(6)]],
    constant long& in_channels [[buffer(7)]],
    constant long& out_channels [[buffer(8)]],
    constant long& kernel_volume [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    #pragma METAL fp math_mode(safe)
    const ulong scalar = ulong(tid);
    if (scalar >= ulong(input_count) * ulong(in_channels)) return;
    const ulong row = scalar / ulong(in_channels);
    const ulong channel = scalar % ulong(in_channels);
    float total = 0.0f;
    for (long pair = input_ptr[row]; pair < input_ptr[row + 1]; ++pair) {
        const ulong output_row = ulong(input_outputs[pair]);
        const ulong offset = ulong(input_offsets[pair]);
        for (ulong out_channel = 0; out_channel < ulong(out_channels); ++out_channel) {
            const ulong weight_index =
                (out_channel * ulong(in_channels) + channel) * ulong(kernel_volume) + offset;
            total = fma(grad_output[output_row * ulong(out_channels) + out_channel],
                        weights[weight_index], total);
        }
    }
    grad_features[scalar] = total;
}
