#include <metal_stdlib>
using namespace metal;

// The stable CSR point order gives each output scalar one writer. Position
// and feature channels share this dispatch, without float atomics or an
// intermediate (N, D + C) tensor.
kernel void voxel_pool_f32(
    device const float* pos [[buffer(0)]],
    device const float* features [[buffer(1)]],
    device const long* point_order [[buffer(2)]],
    device const long* ptr [[buffer(3)]],
    device float* out_pos [[buffer(4)]],
    device float* out_features [[buffer(5)]],
    constant long& voxel_count [[buffer(6)]],
    constant long& dimensions [[buffer(7)]],
    constant long& channels [[buffer(8)]],
    constant int& mean_features [[buffer(9)]],
    uint thread_id [[thread_position_in_grid]])
{
    #pragma METAL fp math_mode(safe)
    const ulong width = ulong(dimensions + channels);
    const ulong output = ulong(thread_id);
    if (output >= ulong(voxel_count) * width) return;
    const ulong row = output / width;
    const ulong column = output % width;
    const bool position = column < ulong(dimensions);
    const ulong source_stride = position ? ulong(dimensions) : ulong(channels);
    const ulong source_column = position ? column : column - ulong(dimensions);
    device const float* source = position ? pos : features;

    // Neumaier's correction keeps small contributions when a dense voxel
    // contains large positive and negative values. The correction changes
    // the float result from index_add_'s atomic order, by design.
    float total = 0.0f;
    float correction = 0.0f;
    const long begin = ptr[row];
    const long end = ptr[row + 1];
    for (long offset = begin; offset < end; ++offset) {
        const ulong point = ulong(point_order[offset]);
        const float value = source[point * source_stride + source_column];
        const float next = total + value;
        if (fabs(total) >= fabs(value)) correction += (total - next) + value;
        else correction += (value - next) + total;
        total = next;
    }
    total += correction;
    if (position || mean_features != 0) total /= float(end - begin);
    if (position) out_pos[row * ulong(dimensions) + column] = total;
    else out_features[row * ulong(channels) + source_column] = total;
}
