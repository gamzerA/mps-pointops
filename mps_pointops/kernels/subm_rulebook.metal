#include <metal_stdlib>
using namespace metal;

// PyTorch 2.7 MPS index_select/gather can corrupt int32 values near
// INT32_MAX. Read the coordinate field directly for the stable sort instead.
kernel void subm_rulebook_gather_axis_i32(
    device const int* indices [[buffer(0)]],
    device const long* sorted_rows [[buffer(1)]],
    device int* axis_values [[buffer(2)]],
    constant long& row_count [[buffer(3)]],
    constant int& axis [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (ulong(tid) >= ulong(row_count)) return;
    axis_values[tid] = indices[4 * ulong(sorted_rows[tid]) + ulong(axis)];
}

// One (kernel offset, output row) per thread. sorted_rows is a device-side
// lexicographic permutation of the four signed int32 coordinate fields.
// Unique coordinates make the binary-search result unambiguous.
kernel void subm_rulebook_lookup_i32(
    device const int* indices [[buffer(0)]],
    device const long* sorted_rows [[buffer(1)]],
    device long* dense_sources [[buffer(2)]],
    device int* valid [[buffer(3)]],
    constant long& row_count [[buffer(4)]],
    constant int& kernel0 [[buffer(5)]],
    constant int& kernel1 [[buffer(6)]],
    constant int& kernel2 [[buffer(7)]],
    constant int& dilation0 [[buffer(8)]],
    constant int& dilation1 [[buffer(9)]],
    constant int& dilation2 [[buffer(10)]],
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
    // All parameters fit int32, so even (K/2)*dilation plus a coordinate
    // remains inside signed int64 before the explicit int32 range check.
    const long target0 = long(indices[base + 1]) + a0 * long(dilation0);
    const long target1 = long(indices[base + 2]) + a1 * long(dilation1);
    const long target2 = long(indices[base + 3]) + a2 * long(dilation2);
    long source = -1;
    // An out-of-range target cannot equal any int32 input. The widening
    // also prevents an addition near INT32_MAX/MIN from wrapping into a
    // different, valid coordinate before the search.
    constexpr long min_i32 = -2147483647L - 1L;
    constexpr long max_i32 = 2147483647L;
    if (target0 >= min_i32 && target0 <= max_i32
        && target1 >= min_i32 && target1 <= max_i32
        && target2 >= min_i32 && target2 <= max_i32) {
        ulong lower = 0;
        ulong upper = ulong(row_count);
        while (lower < upper) {
            const ulong middle = lower + (upper - lower) / 2;
            const long row = sorted_rows[middle];
            const ulong candidate = 4 * ulong(row);
            const int c0 = indices[candidate];
            const int c1 = indices[candidate + 1];
            const int c2 = indices[candidate + 2];
            const int c3 = indices[candidate + 3];
            const bool less = c0 < batch
                || (c0 == batch && (long(c1) < target0
                || (long(c1) == target0 && (long(c2) < target1
                || (long(c2) == target1 && long(c3) < target2)))));
            if (less) lower = middle + 1;
            else upper = middle;
        }
        if (lower < ulong(row_count)) {
            const long row = sorted_rows[lower];
            const ulong candidate = 4 * ulong(row);
            if (indices[candidate] == batch
                && long(indices[candidate + 1]) == target0
                && long(indices[candidate + 2]) == target1
                && long(indices[candidate + 3]) == target2) {
                source = row;
            }
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
