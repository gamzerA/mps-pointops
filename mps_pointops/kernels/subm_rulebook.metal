#include <metal_stdlib>
using namespace metal;

// Bounded reference construction: one (kernel offset, output row) per thread.
// Every lookup scans input rows in their original order. The Python wrapper
// requires unique, valid coordinates; no atomic table or scheduling decision
// can change which row is selected.
kernel void subm_rulebook_lookup_i32(
    device const int* indices [[buffer(0)]],
    device long* dense_sources [[buffer(1)]],
    device int* valid [[buffer(2)]],
    constant long& row_count [[buffer(3)]],
    constant int& kernel0 [[buffer(4)]],
    constant int& kernel1 [[buffer(5)]],
    constant int& kernel2 [[buffer(6)]],
    constant int& dilation0 [[buffer(7)]],
    constant int& dilation1 [[buffer(8)]],
    constant int& dilation2 [[buffer(9)]],
    uint tid [[thread_position_in_grid]])
{
    const ulong volume = ulong(kernel0) * ulong(kernel1) * ulong(kernel2);
    const ulong slots = volume * ulong(row_count);
    if (ulong(tid) >= slots) return;
    const ulong offset = ulong(tid) / ulong(row_count);
    const ulong output = ulong(tid) % ulong(row_count);
    const long a2 = long(offset % ulong(kernel2)) - long(kernel2 / 2);
    const long a1 = long((offset / ulong(kernel2)) % ulong(kernel1)) - long(kernel1 / 2);
    const long a0 = long(offset / (ulong(kernel1) * ulong(kernel2))) - long(kernel0 / 2);
    const ulong base = 4 * output;
    const int batch = indices[base];
    const long target0 = long(indices[base + 1]) + a0 * long(dilation0);
    const long target1 = long(indices[base + 2]) + a1 * long(dilation1);
    const long target2 = long(indices[base + 3]) + a2 * long(dilation2);
    long source = -1;
    for (long row = 0; row < row_count; ++row) {
        const ulong candidate = 4 * ulong(row);
        if (indices[candidate] == batch
            && long(indices[candidate + 1]) == target0
            && long(indices[candidate + 2]) == target1
            && long(indices[candidate + 3]) == target2) {
            source = row;
            break;
        }
    }
    dense_sources[tid] = source;
    valid[tid] = source >= 0 ? 1 : 0;
}

// Both scans are MPS-device int64 inclusive scans. offset_prefix is in
// (offset, output) order; output_prefix is in (output, offset) order. Thus
// the scatter destinations are unique, and padded storage stays initialized
// to -1 by the wrapper. No output size is read back to the host.
kernel void subm_rulebook_compact_i64(
    device const long* dense_sources [[buffer(0)]],
    device const long* offset_prefix [[buffer(1)]],
    device const long* output_prefix [[buffer(2)]],
    device long* pairs [[buffer(3)]],
    device long* output_ptr [[buffer(4)]],
    device long* output_sources [[buffer(5)]],
    device long* output_offsets [[buffer(6)]],
    constant long& row_count [[buffer(7)]],
    constant long& kernel_volume [[buffer(8)]],
    uint tid [[thread_position_in_grid]])
{
    const ulong rows = ulong(row_count);
    const ulong volume = ulong(kernel_volume);
    const ulong slots = rows * volume;
    const ulong work = max(slots, rows + 1);
    if (ulong(tid) >= work) return;
    if (ulong(tid) <= rows) {
        output_ptr[tid] = tid == 0 ? 0 : output_prefix[ulong(tid) * volume - 1];
    }
    if (ulong(tid) >= slots) return;

    const ulong offset = ulong(tid) / rows;
    const ulong output = ulong(tid) % rows;
    const long source = dense_sources[tid];
    if (source >= 0) {
        const ulong rank = ulong(offset_prefix[tid] - 1);
        pairs[3 * rank] = long(offset);
        pairs[3 * rank + 1] = source;
        pairs[3 * rank + 2] = long(output);
    }

    const ulong output_row = ulong(tid) / volume;
    const ulong output_offset = ulong(tid) % volume;
    const long output_source = dense_sources[output_offset * rows + output_row];
    if (output_source >= 0) {
        const ulong rank = ulong(output_prefix[tid] - 1);
        output_sources[rank] = output_source;
        output_offsets[rank] = long(output_offset);
    }
}
