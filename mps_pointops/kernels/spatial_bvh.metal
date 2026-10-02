// SPDX-License-Identifier: Apache-2.0
// The radius arithmetic below adapts this project's MIT-licensed Ball Query
// numerical helper. Its preserved notice is LICENSES/MIT-ball-query.txt.
// Experimental two-level balanced BVH over Morton-sorted 128-point leaves.
// Safe Math only; no public operator.
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

constant constexpr uint BRICK_SIZE = 128;
constant constexpr uint MICRO_LEAVES = 64;
constant constexpr uint MAX_K = 32;
constant constexpr uint STACK_CAPACITY = 32;

static inline bool before(float d, uint i, float e, uint j) {
    return d < e || (d == e && i < j);
}

static inline float squared_norm(float3 d) {
    float s = d.x * d.x;
    s = s + d.y * d.y;
    return s + d.z * d.z;
}

// Match the dense Ball Query's explicit accumulation order. A separate
// multiplication/addition lower bound must not prune against this FMA metric.
static inline float radius_squared_norm(float3 d) {
    float s = d.x * d.x;
    s = fma(d.y, d.y, s);
    return fma(d.z, d.z, s);
}

static inline float lower_bound_sq(float3 q, float3 lo, float3 hi) {
    const float3 d = max(max(lo - q, q - hi), float3(0.0f));
    return squared_norm(d);
}

static inline float node_bound(
    float3 q, device const float *bounds, uint node) {
    const ulong off = ulong(node) * 6;
    const float3 lo = float3(bounds[off], bounds[off + 1], bounds[off + 2]);
    const float3 hi = float3(bounds[off + 3], bounds[off + 4], bounds[off + 5]);
    return lower_bound_sq(q, lo, hi);
}

static inline float radius_node_bound(
    float3 q, device const float *bounds, uint node) {
    const ulong off = ulong(node) * 6;
    const float3 lo = float3(bounds[off], bounds[off + 1], bounds[off + 2]);
    const float3 hi = float3(bounds[off + 3], bounds[off + 4], bounds[off + 5]);
    const float3 gap = max(max(lo - q, q - hi), float3(0.0f));
    return radius_squared_norm(gap);
}

static inline void write_node(
    device float *bounds, device uint *min_index, uint node,
    float3 lo, float3 hi, uint first_index) {
    const ulong off = ulong(node) * 6;
    bounds[off] = lo.x;
    bounds[off + 1] = lo.y;
    bounds[off + 2] = lo.z;
    bounds[off + 3] = hi.x;
    bounds[off + 4] = hi.y;
    bounds[off + 5] = hi.z;
    min_index[node] = first_index;
}

// One 64-lane threadgroup builds 64 leaves and the whole microtree above it.
kernel void bvh_build_micro_f32(
    device const float *points [[buffer(0)]],
    device const long *sorted_indices [[buffer(1)]],
    device float *bounds [[buffer(2)]],
    device uint *min_index [[buffer(3)]],
    constant long &point_count [[buffer(4)]],
    constant uint &padded_leaves [[buffer(5)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float3 local_lo[2 * MICRO_LEAVES];
    threadgroup float3 local_hi[2 * MICRO_LEAVES];
    threadgroup uint local_min[2 * MICRO_LEAVES];
    const uint micro_root = padded_leaves / MICRO_LEAVES + group.x;
    const uint leaf = group.x * MICRO_LEAVES + lane;
    const ulong begin = ulong(leaf) * BRICK_SIZE;
    const ulong end = min(begin + BRICK_SIZE, ulong(point_count));
    float3 lo = float3(INFINITY);
    float3 hi = float3(-INFINITY);
    uint first = UINT_MAX;
    for (ulong p = begin; p < end; ++p) {
        const uint source = uint(sorted_indices[p]);
        const ulong off = ulong(source) * 3;
        const float3 x = float3(points[off], points[off + 1], points[off + 2]);
        lo = min(lo, x);
        hi = max(hi, x);
        first = min(first, source);
    }
    const uint local_leaf = MICRO_LEAVES + lane;
    local_lo[local_leaf] = lo;
    local_hi[local_leaf] = hi;
    local_min[local_leaf] = first;
    write_node(bounds, min_index, padded_leaves + leaf, lo, hi, first);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int level = 5; level >= 0; --level) {
        const uint width = 1u << uint(level);
        if (lane < width) {
            const uint slot = width + lane;
            const uint left = slot * 2;
            const uint right = left + 1;
            lo = min(local_lo[left], local_lo[right]);
            hi = max(local_hi[left], local_hi[right]);
            first = min(local_min[left], local_min[right]);
            local_lo[slot] = lo;
            local_hi[slot] = hi;
            local_min[slot] = first;
            write_node(bounds, min_index, (micro_root << uint(level)) + lane,
                       lo, hi, first);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Macro nodes are only 1..(padded_leaves/64-1), at most 127 for N<=1M.
kernel void bvh_build_macro_f32(
    device float *bounds [[buffer(0)]],
    device uint *min_index [[buffer(1)]],
    constant uint &macro_leaves [[buffer(2)]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float3 local_lo[256];
    threadgroup float3 local_hi[256];
    threadgroup uint local_min[256];
    if (lane < macro_leaves) {
        const uint node = macro_leaves + lane;
        const ulong off = ulong(node) * 6;
        local_lo[node] = float3(bounds[off], bounds[off + 1], bounds[off + 2]);
        local_hi[node] = float3(bounds[off + 3], bounds[off + 4], bounds[off + 5]);
        local_min[node] = min_index[node];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = macro_leaves >> 1; width != 0; width >>= 1) {
        if (lane < width) {
            const uint node = width + lane;
            const uint left = node * 2;
            const uint right = left + 1;
            const float3 lo = min(local_lo[left], local_lo[right]);
            const float3 hi = max(local_hi[left], local_hi[right]);
            const uint first = min(local_min[left], local_min[right]);
            local_lo[node] = lo;
            local_hi[node] = hi;
            local_min[node] = first;
            write_node(bounds, min_index, node, lo, hi, first);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

static inline void insert_candidate(
    float distance, uint source, thread float *best_d, thread uint *best_i,
    uint k, thread uint &found) {
    // A finite coordinate difference can square to +inf. The native dense
    // kNN still ranks that candidate ahead of the (inf, UINT_MAX) sentinel.
    if (isnan(distance) || !before(distance, source, best_d[k - 1], best_i[k - 1])) return;
    uint position = 0;
    while (position < found && before(best_d[position], best_i[position], distance, source)) ++position;
    if (position >= k) return;
    const uint next = min(k, found + 1);
    for (uint s = next - 1; s > position; --s) {
        best_d[s] = best_d[s - 1];
        best_i[s] = best_i[s - 1];
    }
    best_d[position] = distance;
    best_i[position] = source;
    found = next;
}

static inline void scan_point(
    float3 q, uint source, device const float *points,
    thread float *best_d, thread uint *best_i, uint k, thread uint &found) {
    const ulong off = ulong(source) * 3;
    const float3 x = float3(points[off], points[off + 1], points[off + 2]);
    insert_candidate(squared_norm(x - q), source, best_d, best_i, k, found);
}

static inline uint audit_node(
    float3 q, uint node, uint padded_leaves, ulong point_count,
    device const float *points, device const long *sorted_indices,
    float worst_d, uint worst_i) {
    // This debug-only scan checks a prune against the best pair at prune time.
    const uint depth = 31u - clz(node);
    const uint first_at_level = 1u << depth;
    const uint leaf_span = padded_leaves >> depth;
    const ulong begin = ulong(node - first_at_level) * leaf_span * BRICK_SIZE;
    const ulong end = min(begin + ulong(leaf_span) * BRICK_SIZE, point_count);
    uint bad = 0;
    for (ulong p = begin; p < end; ++p) {
        const uint source = uint(sorted_indices[p]);
        const ulong off = ulong(source) * 3;
        const float3 x = float3(points[off], points[off + 1], points[off + 2]);
        const float d = squared_norm(x - q);
        bad += uint(!isnan(d) && before(d, source, worst_d, worst_i));
    }
    return bad;
}

// One scalar thread per query: simple exact prototype, not an occupancy-tuned path.
kernel void bvh_knn_f32(
    device const float *query [[buffer(0)]],
    device const float *points [[buffer(1)]],
    device const long *sorted_indices [[buffer(2)]],
    device const float *bounds [[buffer(3)]],
    device const uint *min_index [[buffer(4)]],
    device float *out_dist [[buffer(5)]],
    device long *out_idx [[buffer(6)]],
    device uint *stats [[buffer(7)]],
    constant long &query_count [[buffer(8)]],
    constant long &point_count [[buffer(9)]],
    constant uint &padded_leaves [[buffer(10)]],
    constant long &K [[buffer(11)]],
    constant uint &audit_enabled [[buffer(12)]],
    uint qi [[thread_position_in_grid]]) {
    if (ulong(qi) >= ulong(query_count)) return;
    const uint k = uint(K);
    const ulong qoff = ulong(qi) * 3;
    const float3 q = float3(query[qoff], query[qoff + 1], query[qoff + 2]);
    float best_d[MAX_K];
    uint best_i[MAX_K];
    for (uint j = 0; j < k; ++j) { best_d[j] = INFINITY; best_i[j] = UINT_MAX; }
    uint found = 0;
    uint visited_nodes = 0;
    uint visited_points = 0;
    uint pruned_nodes = 0;
    uint bad_prunes = 0;
    uint overflow = 0;
    uint stack[STACK_CAPACITY];
    uint top = 0;
    if (all(isfinite(q)) && point_count > 0) stack[top++] = 1;
    while (top != 0) {
        const uint node = stack[--top];
        if (min_index[node] == UINT_MAX) continue;
        ++visited_nodes;
        const float lb = node_bound(q, bounds, node);
        if (found == k && lb > best_d[k - 1]) {
            ++pruned_nodes;
            if (audit_enabled != 0) {
                bad_prunes += audit_node(q, node, padded_leaves, ulong(point_count),
                                          points, sorted_indices, best_d[k - 1], best_i[k - 1]);
            }
            continue;
        }
        if (node >= padded_leaves) {
            const ulong leaf = ulong(node - padded_leaves);
            const ulong begin = leaf * BRICK_SIZE;
            const ulong end = min(begin + BRICK_SIZE, ulong(point_count));
            for (ulong p = begin; p < end; ++p) {
                ++visited_points;
                scan_point(q, uint(sorted_indices[p]), points, best_d, best_i, k, found);
            }
            continue;
        }
        const uint left = node * 2;
        const uint right = left + 1;
        const bool has_left = min_index[left] != UINT_MAX;
        const bool has_right = min_index[right] != UINT_MAX;
        const uint needed = uint(has_left) + uint(has_right);
        if (top + needed > STACK_CAPACITY) { overflow = 1; break; }
        if (has_left && has_right) {
            const float l = node_bound(q, bounds, left);
            const float r = node_bound(q, bounds, right);
            if (l <= r) { stack[top++] = right; stack[top++] = left; }
            else { stack[top++] = left; stack[top++] = right; }
        } else if (has_left) stack[top++] = left;
        else if (has_right) stack[top++] = right;
    }
    if (overflow != 0) {
        // Never return a partial row; the visible status records the fallback.
        found = 0;
        for (uint j = 0; j < k; ++j) { best_d[j] = INFINITY; best_i[j] = UINT_MAX; }
        for (ulong j = 0; j < ulong(point_count); ++j) {
            ++visited_points;
            scan_point(q, uint(j), points, best_d, best_i, k, found);
        }
    }
    const ulong row = ulong(qi) * k;
    for (uint j = 0; j < k; ++j) {
        out_dist[row + j] = best_d[j];
        out_idx[row + j] = best_i[j] == UINT_MAX ? -1 : long(best_i[j]);
    }
    const ulong soff = ulong(qi) * 5;
    stats[soff] = visited_nodes;
    stats[soff + 1] = visited_points;
    stats[soff + 2] = pruned_nodes;
    stats[soff + 3] = bad_prunes;
    stats[soff + 4] = overflow;
}

static inline bool radius_candidate(
    float3 q, uint source, device const float *points,
    float radius, float radius_sq, thread float &distance_sq) {
    const ulong off = ulong(source) * 3;
    const float3 x = float3(points[off], points[off + 1], points[off + 2]);
    const float3 delta = q - x;
    if (!all(isfinite(x)) || !all(isfinite(delta))) return false;
    if (radius < 0x1p-50f || !isfinite(radius_sq)) {
        const float normalized_sq = radius_squared_norm(delta / radius);
        if (!(normalized_sq < 1.0f)) return false;
        distance_sq = (radius_sq >= 0x1p-126f && isfinite(radius_sq))
            ? normalized_sq * radius_sq : radius_squared_norm(delta);
        return true;
    }
    distance_sq = radius_squared_norm(delta);
    return distance_sq < radius_sq;
}

static inline void radius_insert(
    float distance_sq, uint source,
    thread float *best_d, thread uint *best_i,
    uint k, thread uint &found) {
    if (found == k && source >= best_i[k - 1]) return;
    uint position = 0;
    while (position < found && best_i[position] < source) ++position;
    if (position >= k) return;
    const uint next = min(k, found + 1);
    for (uint s = next - 1; s > position; --s) {
        best_d[s] = best_d[s - 1];
        best_i[s] = best_i[s - 1];
    }
    best_d[position] = distance_sq;
    best_i[position] = source;
    found = next;
}

static inline uint audit_radius_node(
    float3 q, uint node, uint padded_leaves, ulong point_count,
    device const float *points, device const long *sorted_indices,
    float radius, float radius_sq, uint worst_index) {
    const uint depth = 31u - clz(node);
    const uint first_at_level = 1u << depth;
    const uint leaf_span = padded_leaves >> depth;
    const ulong begin = ulong(node - first_at_level) * leaf_span * BRICK_SIZE;
    const ulong end = min(begin + ulong(leaf_span) * BRICK_SIZE, point_count);
    uint missed = 0;
    for (ulong p = begin; p < end; ++p) {
        const uint source = uint(sorted_indices[p]);
        if (source >= worst_index) continue;
        float distance_sq = 0.0f;
        missed += uint(radius_candidate(q, source, points, radius, radius_sq, distance_sq));
    }
    return missed;
}

// Research-only, single-cloud strict-radius Ball Query. Each query owns a row;
// candidates are kept by original source index, independent of Morton order.
// For r >= 2^-50 and finite r^2, source-level monotonicity of the identical
// FMA metric makes the AABB gap a lower bound on every candidate's distance.
// Prune only for a strict lb > r^2. Tiny/overflowing radii use index-only
// pruning, avoiding any geometric decision in the normalized-distance path.
kernel void bvh_radius_f32(
    device const float *query [[buffer(0)]],
    device const float *points [[buffer(1)]],
    device const long *sorted_indices [[buffer(2)]],
    device const float *bounds [[buffer(3)]],
    device const uint *min_index [[buffer(4)]],
    device float *out_dist [[buffer(5)]],
    device long *out_idx [[buffer(6)]],
    device uint *stats [[buffer(7)]],
    constant long &query_count [[buffer(8)]],
    constant long &point_count [[buffer(9)]],
    constant uint &padded_leaves [[buffer(10)]],
    constant long &K [[buffer(11)]],
    constant float &radius [[buffer(12)]],
    constant float &radius_sq [[buffer(13)]],
    constant uint &audit_enabled [[buffer(14)]],
    uint qi [[thread_position_in_grid]]) {
    if (ulong(qi) >= ulong(query_count)) return;
    const uint k = uint(K);
    const ulong qoff = ulong(qi) * 3;
    const float3 q = float3(query[qoff], query[qoff + 1], query[qoff + 2]);
    float best_d[MAX_K];
    uint best_i[MAX_K];
    for (uint j = 0; j < k; ++j) { best_d[j] = 0.0f; best_i[j] = UINT_MAX; }
    uint found = 0;
    uint visited_nodes = 0;
    uint visited_points = 0;
    uint pruned_nodes = 0;
    uint bad_prunes = 0;
    uint overflow = 0;
    uint stack[STACK_CAPACITY];
    uint top = 0;
    if (all(isfinite(q)) && point_count > 0 && radius > 0.0f) stack[top++] = 1;
    while (top != 0) {
        const uint node = stack[--top];
        if (min_index[node] == UINT_MAX) continue;
        ++visited_nodes;
        const bool index_prune = found == k && min_index[node] >= best_i[k - 1];
        const bool geometric_prune = !index_prune &&
            radius >= 0x1p-50f && isfinite(radius_sq) &&
            radius_node_bound(q, bounds, node) > radius_sq;
        if (index_prune || geometric_prune) {
            ++pruned_nodes;
            if (audit_enabled != 0) {
                bad_prunes += audit_radius_node(
                    q, node, padded_leaves, ulong(point_count), points,
                    sorted_indices, radius, radius_sq,
                    found == k ? best_i[k - 1] : UINT_MAX);
            }
            continue;
        }
        if (node >= padded_leaves) {
            const ulong begin = ulong(node - padded_leaves) * BRICK_SIZE;
            const ulong end = min(begin + BRICK_SIZE, ulong(point_count));
            for (ulong p = begin; p < end; ++p) {
                ++visited_points;
                const uint source = uint(sorted_indices[p]);
                if (found == k && source >= best_i[k - 1]) continue;
                float distance_sq = 0.0f;
                if (radius_candidate(q, source, points, radius, radius_sq, distance_sq)) {
                    radius_insert(distance_sq, source, best_d, best_i, k, found);
                }
            }
            continue;
        }
        const uint left = node * 2;
        const uint right = left + 1;
        const bool has_left = min_index[left] != UINT_MAX;
        const bool has_right = min_index[right] != UINT_MAX;
        const uint needed = uint(has_left) + uint(has_right);
        if (top + needed > STACK_CAPACITY) { overflow = 1; break; }
        // Visit the child whose subtree has the earlier source index first.
        if (has_left && has_right) {
            if (min_index[left] <= min_index[right]) {
                stack[top++] = right; stack[top++] = left;
            } else {
                stack[top++] = left; stack[top++] = right;
            }
        } else if (has_left) stack[top++] = left;
        else if (has_right) stack[top++] = right;
    }
    if (overflow != 0) {
        found = 0;
        for (uint j = 0; j < k; ++j) { best_d[j] = 0.0f; best_i[j] = UINT_MAX; }
        for (ulong j = 0; j < ulong(point_count); ++j) {
            ++visited_points;
            float distance_sq = 0.0f;
            if (radius_candidate(q, uint(j), points, radius, radius_sq, distance_sq)) {
                radius_insert(distance_sq, uint(j), best_d, best_i, k, found);
            }
        }
    }
    const ulong row = ulong(qi) * k;
    for (uint j = 0; j < k; ++j) {
        out_dist[row + j] = best_d[j];
        out_idx[row + j] = best_i[j] == UINT_MAX ? -1 : long(best_i[j]);
    }
    const ulong soff = ulong(qi) * 5;
    stats[soff] = visited_nodes;
    stats[soff + 1] = visited_points;
    stats[soff + 2] = pruned_nodes;
    stats[soff + 3] = bad_prunes;
    stats[soff + 4] = overflow;
}

// Optional low-Q path: a cheap nearest-box leaf supplies a valid upper bound
// on the global Kth distance. Every microtree is then searched independently.
kernel void bvh_seed_f32(
    device const float *query [[buffer(0)]],
    device const float *points [[buffer(1)]],
    device const long *sorted_indices [[buffer(2)]],
    device const float *bounds [[buffer(3)]],
    device const uint *min_index [[buffer(4)]],
    device float *seed_threshold [[buffer(5)]],
    constant long &query_count [[buffer(6)]],
    constant long &point_count [[buffer(7)]],
    constant uint &padded_leaves [[buffer(8)]],
    constant long &K [[buffer(9)]],
    uint qi [[thread_position_in_grid]]) {
    if (ulong(qi) >= ulong(query_count)) return;
    const uint k = uint(K);
    const ulong qoff = ulong(qi) * 3;
    const float3 q = float3(query[qoff], query[qoff + 1], query[qoff + 2]);
    if (!all(isfinite(q)) || point_count == 0) { seed_threshold[qi] = INFINITY; return; }
    uint node = 1;
    while (node < padded_leaves) {
        const uint left = node * 2;
        const uint right = left + 1;
        if (min_index[left] == UINT_MAX) node = right;
        else if (min_index[right] == UINT_MAX) node = left;
        else node = node_bound(q, bounds, left) <= node_bound(q, bounds, right) ? left : right;
    }
    float best_d[MAX_K];
    uint best_i[MAX_K];
    for (uint j = 0; j < k; ++j) { best_d[j] = INFINITY; best_i[j] = UINT_MAX; }
    uint found = 0;
    const ulong begin = ulong(node - padded_leaves) * BRICK_SIZE;
    const ulong end = min(begin + BRICK_SIZE, ulong(point_count));
    for (ulong p = begin; p < end; ++p) {
        scan_point(q, uint(sorted_indices[p]), points, best_d, best_i, k, found);
    }
    seed_threshold[qi] = found == k ? best_d[k - 1] : INFINITY;
}

kernel void bvh_search_micro_f32(
    device const float *query [[buffer(0)]],
    device const float *points [[buffer(1)]],
    device const long *sorted_indices [[buffer(2)]],
    device const float *bounds [[buffer(3)]],
    device const uint *min_index [[buffer(4)]],
    device const float *seed_threshold [[buffer(5)]],
    device float *local_dist [[buffer(6)]],
    device uint *local_idx [[buffer(7)]],
    device uint *local_count [[buffer(8)]],
    device uint *local_stats [[buffer(9)]],
    constant long &query_count [[buffer(10)]],
    constant long &point_count [[buffer(11)]],
    constant uint &padded_leaves [[buffer(12)]],
    constant long &K [[buffer(13)]],
    uint task [[thread_position_in_grid]]) {
    const uint microtrees = padded_leaves / MICRO_LEAVES;
    const ulong task_count = ulong(query_count) * microtrees;
    if (ulong(task) >= task_count) return;
    const uint qi = task / microtrees;
    const uint micro = task % microtrees;
    const ulong qoff = ulong(qi) * 3;
    const float3 q = float3(query[qoff], query[qoff + 1], query[qoff + 2]);
    const uint k = uint(K);
    float best_d[MAX_K];
    uint best_i[MAX_K];
    for (uint j = 0; j < k; ++j) { best_d[j] = INFINITY; best_i[j] = UINT_MAX; }
    uint found = 0;
    uint visited_nodes = 0;
    uint visited_points = 0;
    uint pruned_nodes = 0;
    uint overflow = 0;
    uint stack[16];
    uint top = 0;
    const uint root = microtrees + micro;
    if (all(isfinite(q)) && min_index[root] != UINT_MAX) stack[top++] = root;
    while (top != 0) {
        const uint node = stack[--top];
        ++visited_nodes;
        const float lb = node_bound(q, bounds, node);
        if (lb > seed_threshold[qi] || (found == k && lb > best_d[k - 1])) {
            ++pruned_nodes;
            continue;
        }
        if (node >= padded_leaves) {
            const ulong begin = ulong(node - padded_leaves) * BRICK_SIZE;
            const ulong end = min(begin + BRICK_SIZE, ulong(point_count));
            for (ulong p = begin; p < end; ++p) {
                ++visited_points;
                scan_point(q, uint(sorted_indices[p]), points, best_d, best_i, k, found);
            }
            continue;
        }
        const uint left = node * 2;
        const uint right = left + 1;
        const bool has_left = min_index[left] != UINT_MAX;
        const bool has_right = min_index[right] != UINT_MAX;
        if (top + uint(has_left) + uint(has_right) > 16) { overflow = 1; break; }
        if (has_left && has_right) {
            const float l = node_bound(q, bounds, left);
            const float r = node_bound(q, bounds, right);
            if (l <= r) { stack[top++] = right; stack[top++] = left; }
            else { stack[top++] = left; stack[top++] = right; }
        } else if (has_left) stack[top++] = left;
        else if (has_right) stack[top++] = right;
    }
    if (overflow != 0) {
        // The local result must represent this entire microtree before merge.
        found = 0;
        for (uint j = 0; j < k; ++j) { best_d[j] = INFINITY; best_i[j] = UINT_MAX; }
        const ulong begin = ulong(micro) * MICRO_LEAVES * BRICK_SIZE;
        const ulong end = min(begin + MICRO_LEAVES * BRICK_SIZE, ulong(point_count));
        for (ulong p = begin; p < end; ++p) {
            ++visited_points;
            scan_point(q, uint(sorted_indices[p]), points, best_d, best_i, k, found);
        }
    }
    const ulong base = ulong(task) * k;
    for (uint j = 0; j < k; ++j) {
        local_dist[base + j] = best_d[j];
        local_idx[base + j] = best_i[j];
    }
    local_count[task] = found;
    const ulong soff = ulong(task) * 4;
    local_stats[soff] = visited_nodes;
    local_stats[soff + 1] = visited_points;
    local_stats[soff + 2] = pruned_nodes;
    local_stats[soff + 3] = overflow;
}

kernel void bvh_merge_micro_f32(
    device const float *local_dist [[buffer(0)]],
    device const uint *local_idx [[buffer(1)]],
    device const uint *local_count [[buffer(2)]],
    device const uint *local_stats [[buffer(3)]],
    device float *out_dist [[buffer(4)]],
    device long *out_idx [[buffer(5)]],
    device uint *stats [[buffer(6)]],
    constant long &query_count [[buffer(7)]],
    constant uint &microtrees [[buffer(8)]],
    constant long &K [[buffer(9)]],
    uint qi [[thread_position_in_grid]]) {
    if (ulong(qi) >= ulong(query_count)) return;
    const uint k = uint(K);
    float best_d[MAX_K];
    uint best_i[MAX_K];
    for (uint j = 0; j < k; ++j) { best_d[j] = INFINITY; best_i[j] = UINT_MAX; }
    uint found = 0;
    uint visited_nodes = 0;
    uint visited_points = 0;
    uint pruned_nodes = 0;
    uint overflows = 0;
    for (uint micro = 0; micro < microtrees; ++micro) {
        const ulong task = ulong(qi) * microtrees + micro;
        for (uint j = 0; j < local_count[task]; ++j) {
            const ulong off = task * k + j;
            insert_candidate(local_dist[off], local_idx[off], best_d, best_i, k, found);
        }
        const ulong soff = task * 4;
        visited_nodes += local_stats[soff];
        visited_points += local_stats[soff + 1];
        pruned_nodes += local_stats[soff + 2];
        overflows += local_stats[soff + 3];
    }
    const ulong row = ulong(qi) * k;
    for (uint j = 0; j < k; ++j) {
        out_dist[row + j] = best_d[j];
        out_idx[row + j] = best_i[j] == UINT_MAX ? -1 : long(best_i[j]);
    }
    const ulong soff = ulong(qi) * 5;
    stats[soff] = visited_nodes;
    stats[soff + 1] = visited_points;
    stats[soff + 2] = pruned_nodes;
    stats[soff + 3] = 0;  // Full prune audit is available on the serial path.
    stats[soff + 4] = overflows;
}
