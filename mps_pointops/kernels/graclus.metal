#include <metal_stdlib>
using namespace metal;

// One work item performs all choices in node-permutation order. It writes the
// entire output, including isolated vertices; no prior buffer fill is needed.
kernel void graclus_greedy(
    device const long* rowptr [[buffer(0)]],
    device const long* col [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device const long* node_order [[buffer(3)]],
    device long* labels [[buffer(4)]],
    constant long& num_nodes [[buffer(5)]],
    constant int& weighted [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    for (long i = 0; i < num_nodes; ++i) labels[i] = -1;
    for (long position = 0; position < num_nodes; ++position) {
        const long center = node_order[position];
        if (labels[center] >= 0) continue;
        long chosen = center;
        if (weighted == 0) {
            for (long edge = rowptr[center]; edge < rowptr[center + 1]; ++edge) {
                const long neighbor = col[edge];
                if (labels[neighbor] < 0) {
                    chosen = neighbor;
                    break;
                }
            }
        } else {
            float maximum = 0.0f;
            for (long edge = rowptr[center]; edge < rowptr[center + 1]; ++edge) {
                const long neighbor = col[edge];
                if (labels[neighbor] < 0 && weight[edge] >= maximum) {
                    maximum = weight[edge];
                    chosen = neighbor;
                }
            }
        }
        const long identifier = min(center, chosen);
        labels[center] = identifier;
        labels[chosen] = identifier;
    }
}
