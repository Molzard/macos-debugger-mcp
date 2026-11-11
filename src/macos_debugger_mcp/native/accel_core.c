#include "accel_core.h"
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PAGE_SIZE_ARM64 16384

// Fast 64-byte block equality check using 4 unrolled 128-bit NEON registers
static inline int is_block_identical_64(const uint8_t *a, const uint8_t *b) {
    uint64x2_t o0 = vld1q_u64((const uint64_t *)(a + 0));
    uint64x2_t n0 = vld1q_u64((const uint64_t *)(b + 0));
    uint64x2_t o1 = vld1q_u64((const uint64_t *)(a + 16));
    uint64x2_t n1 = vld1q_u64((const uint64_t *)(b + 16));
    uint64x2_t o2 = vld1q_u64((const uint64_t *)(a + 32));
    uint64x2_t n2 = vld1q_u64((const uint64_t *)(b + 32));
    uint64x2_t o3 = vld1q_u64((const uint64_t *)(a + 48));
    uint64x2_t n3 = vld1q_u64((const uint64_t *)(b + 48));

    uint64x2_t diff0 = veorq_u64(o0, n0);
    uint64x2_t diff1 = veorq_u64(o1, n1);
    uint64x2_t diff2 = veorq_u64(o2, n2);
    uint64x2_t diff3 = veorq_u64(o3, n3);

    uint64x2_t any_diff = vorrq_u64(vorrq_u64(diff0, diff1), vorrq_u64(diff2, diff3));
    return vmaxvq_u32(vreinterpretq_u32_u64(any_diff)) == 0;
}

// -------------------------------------------------------------------------
// Wildcard AoB Scanner with ARM64 NEON Dual-Anchor Filtering
// -------------------------------------------------------------------------

int64_t aob_scan_neon(
    const uint8_t *data,
    size_t data_len,
    const uint8_t *pattern,
    const uint8_t *mask,
    size_t pattern_len
) {
    if (!data || !pattern || !mask || pattern_len == 0 || data_len < pattern_len) {
        return -1;
    }

    // 1. Locate first non-wildcard anchor (head)
    size_t head_idx = 0;
    int has_head = 0;
    for (size_t i = 0; i < pattern_len; ++i) {
        if (mask[i] == 0xFF) {
            head_idx = i;
            has_head = 1;
            break;
        }
    }

    // If entire pattern is wildcards, offset 0 matches trivially
    if (!has_head) {
        return 0;
    }

    uint8_t head_byte = pattern[head_idx];

    // 2. Locate last non-wildcard anchor (tail)
    size_t tail_idx = head_idx;
    int has_tail = 0;
    for (size_t i = pattern_len; i > head_idx + 1; --i) {
        if (mask[i - 1] == 0xFF) {
            tail_idx = i - 1;
            has_tail = 1;
            break;
        }
    }

    uint8_t tail_byte = has_tail ? pattern[tail_idx] : 0;
    uint8x16_t v_head = vdupq_n_u8(head_byte);
    uint8x16_t v_tail = has_tail ? vdupq_n_u8(tail_byte) : vdupq_n_u8(0);

    size_t max_scan = data_len - pattern_len;
    size_t max_anchor = has_tail ? tail_idx : head_idx;
    size_t offset = 0;

    // Vectorized 16-byte lane scan
    while (offset + 16 + max_anchor <= data_len && offset <= max_scan) {
        uint8x16_t chunk_head = vld1q_u8(data + offset + head_idx);
        uint8x16_t cmp_head = vceqq_u8(chunk_head, v_head);
        uint8x16_t match_vec = cmp_head;

        if (has_tail) {
            uint8x16_t chunk_tail = vld1q_u8(data + offset + tail_idx);
            uint8x16_t cmp_tail = vceqq_u8(chunk_tail, v_tail);
            match_vec = vandq_u8(cmp_head, cmp_tail);
        }

        // 1-cycle horizontal reduction on Apple Silicon
        if (vmaxvq_u8(match_vec) != 0) {
            uint8_t mask_lanes[16];
            vst1q_u8(mask_lanes, match_vec);

            for (int lane = 0; lane < 16; ++lane) {
                if (mask_lanes[lane]) {
                    size_t cand_offset = offset + (size_t)lane;
                    if (cand_offset > max_scan) break;

                    const uint8_t *cand = data + cand_offset;
                    int match = 1;
                    for (size_t p = 0; p < pattern_len; ++p) {
                        if ((cand[p] & mask[p]) != pattern[p]) {
                            match = 0;
                            break;
                        }
                    }
                    if (match) {
                        return (int64_t)cand_offset;
                    }
                }
            }
        }
        offset += 16;
    }

    // Scalar fallback for remaining tail bytes
    while (offset <= max_scan) {
        const uint8_t *cand = data + offset;
        int match = 1;
        for (size_t p = 0; p < pattern_len; ++p) {
            if ((cand[p] & mask[p]) != pattern[p]) {
                match = 0;
                break;
            }
        }
        if (match) {
            return (int64_t)offset;
        }
        offset++;
    }

    return -1;
}

size_t aob_scan_all_neon(
    const uint8_t *data,
    size_t data_len,
    const uint8_t *pattern,
    const uint8_t *mask,
    size_t pattern_len,
    uint64_t *out_offsets,
    size_t max_out
) {
    if (!data || !pattern || !mask || pattern_len == 0 || data_len < pattern_len || !out_offsets || max_out == 0) {
        return 0;
    }

    size_t head_idx = 0;
    int has_head = 0;
    for (size_t i = 0; i < pattern_len; ++i) {
        if (mask[i] == 0xFF) {
            head_idx = i;
            has_head = 1;
            break;
        }
    }

    if (!has_head) {
        size_t count = 0;
        size_t limit = data_len - pattern_len + 1;
        for (size_t i = 0; i < limit && count < max_out; ++i) {
            out_offsets[count++] = i;
        }
        return count;
    }

    uint8_t head_byte = pattern[head_idx];
    size_t tail_idx = head_idx;
    int has_tail = 0;
    for (size_t i = pattern_len; i > head_idx + 1; --i) {
        if (mask[i - 1] == 0xFF) {
            tail_idx = i - 1;
            has_tail = 1;
            break;
        }
    }

    uint8_t tail_byte = has_tail ? pattern[tail_idx] : 0;
    uint8x16_t v_head = vdupq_n_u8(head_byte);
    uint8x16_t v_tail = has_tail ? vdupq_n_u8(tail_byte) : vdupq_n_u8(0);

    size_t max_scan = data_len - pattern_len;
    size_t max_anchor = has_tail ? tail_idx : head_idx;
    size_t offset = 0;
    size_t found_count = 0;

    while (offset + 16 + max_anchor <= data_len && offset <= max_scan && found_count < max_out) {
        uint8x16_t chunk_head = vld1q_u8(data + offset + head_idx);
        uint8x16_t cmp_head = vceqq_u8(chunk_head, v_head);
        uint8x16_t match_vec = cmp_head;

        if (has_tail) {
            uint8x16_t chunk_tail = vld1q_u8(data + offset + tail_idx);
            uint8x16_t cmp_tail = vceqq_u8(chunk_tail, v_tail);
            match_vec = vandq_u8(cmp_head, cmp_tail);
        }

        if (vmaxvq_u8(match_vec) != 0) {
            uint8_t mask_lanes[16];
            vst1q_u8(mask_lanes, match_vec);

            for (int lane = 0; lane < 16; ++lane) {
                if (mask_lanes[lane]) {
                    size_t cand_offset = offset + (size_t)lane;
                    if (cand_offset > max_scan) break;

                    const uint8_t *cand = data + cand_offset;
                    int match = 1;
                    for (size_t p = 0; p < pattern_len; ++p) {
                        if ((cand[p] & mask[p]) != pattern[p]) {
                            match = 0;
                            break;
                        }
                    }
                    if (match) {
                        out_offsets[found_count++] = (uint64_t)cand_offset;
                        if (found_count >= max_out) return found_count;
                    }
                }
            }
        }
        offset += 16;
    }

    while (offset <= max_scan && found_count < max_out) {
        const uint8_t *cand = data + offset;
        int match = 1;
        for (size_t p = 0; p < pattern_len; ++p) {
            if ((cand[p] & mask[p]) != pattern[p]) {
                match = 0;
                break;
            }
        }
        if (match) {
            out_offsets[found_count++] = (uint64_t)offset;
        }
        offset++;
    }

    return found_count;
}

// -------------------------------------------------------------------------
// Vectorized Snapshot Diffing with 16KB Fast Page-Skip
// -------------------------------------------------------------------------

size_t diff_snapshots_neon(
    const uint8_t *old_buf,
    const uint8_t *new_buf,
    size_t size,
    int comp_type,
    int data_type,
    double target_val,
    uint64_t *out_offsets,
    size_t max_out
) {
    if (!old_buf || !new_buf || size == 0 || !out_offsets || max_out == 0) {
        return 0;
    }

    size_t item_size = 4;
    if (data_type == TYPE_INT64 || data_type == TYPE_UINT64 || data_type == TYPE_DOUBLE) {
        item_size = 8;
    }

    size_t limit = size - (size % item_size);
    size_t count = 0;
    size_t offset = 0;

    // Fast Page-Skip is valid when checking for non-equality / changes
    // If checking for unchanged, identical pages must be processed, not skipped!
    int can_fast_skip = (comp_type != COMP_UNCHANGED);
    if (comp_type == COMP_EXACT_DIFFERENCE && target_val == 0.0) {
        can_fast_skip = 0; // exact difference of 0 is equivalent to unchanged
    }

    while (offset < limit && count < max_out) {
        // Check 16 KB page fast-skip
        if (can_fast_skip && (offset + PAGE_SIZE_ARM64 <= limit)) {
            const uint8_t *p_old = old_buf + offset;
            const uint8_t *p_new = new_buf + offset;
            int page_identical = 1;

            // Unrolled check 256 * 64 bytes = 16384 bytes
            for (size_t b = 0; b < PAGE_SIZE_ARM64; b += 64) {
                if (!is_block_identical_64(p_old + b, p_new + b)) {
                    page_identical = 0;
                    break;
                }
            }

            if (page_identical) {
                offset += PAGE_SIZE_ARM64;
                continue;
            }
        }

        // Process block elements
        switch (data_type) {
            case TYPE_INT32: {
                size_t step = 16; // 4 int32 per vector
                while (offset + step <= limit && count < max_out) {
                    int32x4_t vo = vld1q_s32((const int32_t *)(old_buf + offset));
                    int32x4_t vn = vld1q_s32((const int32_t *)(new_buf + offset));
                    uint32x4_t cmp;

                    switch (comp_type) {
                        case COMP_CHANGED:
                            cmp = vmvnq_u32(vceqq_s32(vo, vn));
                            break;
                        case COMP_UNCHANGED:
                            cmp = vceqq_s32(vo, vn);
                            break;
                        case COMP_INCREASED:
                            cmp = vcgtq_s32(vn, vo);
                            break;
                        case COMP_DECREASED:
                            cmp = vcgtq_s32(vo, vn);
                            break;
                        case COMP_EXACT_DIFFERENCE: {
                            int32x4_t diff = vsubq_s32(vn, vo);
                            int32x4_t target = vdupq_n_s32((int32_t)target_val);
                            cmp = vceqq_s32(diff, target);
                            break;
                        }
                        default:
                            cmp = vdupq_n_u32(0);
                            break;
                    }

                    if (vmaxvq_u32(cmp) != 0) {
                        uint32_t lanes[4];
                        vst1q_u32(lanes, cmp);
                        for (int i = 0; i < 4; ++i) {
                            if (lanes[i]) {
                                out_offsets[count++] = offset + (size_t)i * 4;
                                if (count >= max_out) return count;
                            }
                        }
                    }
                    offset += step;
                }
                break;
            }

            case TYPE_UINT32: {
                size_t step = 16;
                while (offset + step <= limit && count < max_out) {
                    uint32x4_t vo = vld1q_u32((const uint32_t *)(old_buf + offset));
                    uint32x4_t vn = vld1q_u32((const uint32_t *)(new_buf + offset));
                    uint32x4_t cmp;

                    switch (comp_type) {
                        case COMP_CHANGED:
                            cmp = vmvnq_u32(vceqq_u32(vo, vn));
                            break;
                        case COMP_UNCHANGED:
                            cmp = vceqq_u32(vo, vn);
                            break;
                        case COMP_INCREASED:
                            cmp = vcgtq_u32(vn, vo);
                            break;
                        case COMP_DECREASED:
                            cmp = vcgtq_u32(vo, vn);
                            break;
                        case COMP_EXACT_DIFFERENCE: {
                            uint32x4_t diff = vsubq_u32(vn, vo);
                            uint32x4_t target = vdupq_n_u32((uint32_t)target_val);
                            cmp = vceqq_u32(diff, target);
                            break;
                        }
                        default:
                            cmp = vdupq_n_u32(0);
                            break;
                    }

                    if (vmaxvq_u32(cmp) != 0) {
                        uint32_t lanes[4];
                        vst1q_u32(lanes, cmp);
                        for (int i = 0; i < 4; ++i) {
                            if (lanes[i]) {
                                out_offsets[count++] = offset + (size_t)i * 4;
                                if (count >= max_out) return count;
                            }
                        }
                    }
                    offset += step;
                }
                break;
            }

            case TYPE_FLOAT: {
                size_t step = 16;
                while (offset + step <= limit && count < max_out) {
                    float32x4_t vo = vld1q_f32((const float *)(old_buf + offset));
                    float32x4_t vn = vld1q_f32((const float *)(new_buf + offset));
                    uint32x4_t cmp;

                    switch (comp_type) {
                        case COMP_CHANGED:
                            cmp = vmvnq_u32(vceqq_f32(vo, vn));
                            break;
                        case COMP_UNCHANGED:
                            cmp = vceqq_f32(vo, vn);
                            break;
                        case COMP_INCREASED:
                            cmp = vcgtq_f32(vn, vo);
                            break;
                        case COMP_DECREASED:
                            cmp = vcgtq_f32(vo, vn);
                            break;
                        case COMP_EXACT_DIFFERENCE: {
                            float32x4_t diff = vsubq_f32(vn, vo);
                            float32x4_t target = vdupq_n_f32((float)target_val);
                            float32x4_t err = vabsq_f32(vsubq_f32(diff, target));
                            cmp = vcltq_f32(err, vdupq_n_f32(1e-4f));
                            break;
                        }
                        default:
                            cmp = vdupq_n_u32(0);
                            break;
                    }

                    if (vmaxvq_u32(cmp) != 0) {
                        uint32_t lanes[4];
                        vst1q_u32(lanes, cmp);
                        for (int i = 0; i < 4; ++i) {
                            if (lanes[i]) {
                                out_offsets[count++] = offset + (size_t)i * 4;
                                if (count >= max_out) return count;
                            }
                        }
                    }
                    offset += step;
                }
                break;
            }

            case TYPE_INT64: {
                size_t step = 16; // 2 int64 per vector
                while (offset + step <= limit && count < max_out) {
                    int64x2_t vo = vld1q_s64((const int64_t *)(old_buf + offset));
                    int64x2_t vn = vld1q_s64((const int64_t *)(new_buf + offset));
                    uint64x2_t cmp;

                    switch (comp_type) {
                        case COMP_CHANGED:
                            cmp = vmvnq_u8(vreinterpretq_u8_u64(vceqq_s64(vo, vn)));
                            break;
                        case COMP_UNCHANGED:
                            cmp = vceqq_s64(vo, vn);
                            break;
                        case COMP_INCREASED:
                            cmp = vcgtq_s64(vn, vo);
                            break;
                        case COMP_DECREASED:
                            cmp = vcgtq_s64(vo, vn);
                            break;
                        case COMP_EXACT_DIFFERENCE: {
                            int64x2_t diff = vsubq_s64(vn, vo);
                            int64x2_t target = vdupq_n_s64((int64_t)target_val);
                            cmp = vceqq_s64(diff, target);
                            break;
                        }
                        default:
                            cmp = vdupq_n_u64(0);
                            break;
                    }

                    if (vmaxvq_u32(vreinterpretq_u32_u64(cmp)) != 0) {
                        uint64_t lanes[2];
                        vst1q_u64(lanes, cmp);
                        for (int i = 0; i < 2; ++i) {
                            if (lanes[i]) {
                                out_offsets[count++] = offset + (size_t)i * 8;
                                if (count >= max_out) return count;
                            }
                        }
                    }
                    offset += step;
                }
                break;
            }

            case TYPE_UINT64: {
                size_t step = 16;
                while (offset + step <= limit && count < max_out) {
                    uint64x2_t vo = vld1q_u64((const uint64_t *)(old_buf + offset));
                    uint64x2_t vn = vld1q_u64((const uint64_t *)(new_buf + offset));
                    uint64x2_t cmp;

                    switch (comp_type) {
                        case COMP_CHANGED:
                            cmp = vmvnq_u8(vreinterpretq_u8_u64(vceqq_u64(vo, vn)));
                            break;
                        case COMP_UNCHANGED:
                            cmp = vceqq_u64(vo, vn);
                            break;
                        case COMP_INCREASED:
                            cmp = vcgtq_u64(vn, vo);
                            break;
                        case COMP_DECREASED:
                            cmp = vcgtq_u64(vo, vn);
                            break;
                        case COMP_EXACT_DIFFERENCE: {
                            uint64x2_t diff = vsubq_u64(vn, vo);
                            uint64x2_t target = vdupq_n_u64((uint64_t)target_val);
                            cmp = vceqq_u64(diff, target);
                            break;
                        }
                        default:
                            cmp = vdupq_n_u64(0);
                            break;
                    }

                    if (vmaxvq_u32(vreinterpretq_u32_u64(cmp)) != 0) {
                        uint64_t lanes[2];
                        vst1q_u64(lanes, cmp);
                        for (int i = 0; i < 2; ++i) {
                            if (lanes[i]) {
                                out_offsets[count++] = offset + (size_t)i * 8;
                                if (count >= max_out) return count;
                            }
                        }
                    }
                    offset += step;
                }
                break;
            }

            case TYPE_DOUBLE: {
                size_t step = 16;
                while (offset + step <= limit && count < max_out) {
                    float64x2_t vo = vld1q_f64((const double *)(old_buf + offset));
                    float64x2_t vn = vld1q_f64((const double *)(new_buf + offset));
                    uint64x2_t cmp;

                    switch (comp_type) {
                        case COMP_CHANGED:
                            cmp = vmvnq_u8(vreinterpretq_u8_u64(vceqq_f64(vo, vn)));
                            break;
                        case COMP_UNCHANGED:
                            cmp = vceqq_f64(vo, vn);
                            break;
                        case COMP_INCREASED:
                            cmp = vcgtq_f64(vn, vo);
                            break;
                        case COMP_DECREASED:
                            cmp = vcgtq_f64(vo, vn);
                            break;
                        case COMP_EXACT_DIFFERENCE: {
                            float64x2_t diff = vsubq_f64(vn, vo);
                            float64x2_t target = vdupq_n_f64(target_val);
                            float64x2_t err = vabsq_f64(vsubq_f64(diff, target));
                            cmp = vcltq_f64(err, vdupq_n_f64(1e-4));
                            break;
                        }
                        default:
                            cmp = vdupq_n_u64(0);
                            break;
                    }

                    if (vmaxvq_u32(vreinterpretq_u32_u64(cmp)) != 0) {
                        uint64_t lanes[2];
                        vst1q_u64(lanes, cmp);
                        for (int i = 0; i < 2; ++i) {
                            if (lanes[i]) {
                                out_offsets[count++] = offset + (size_t)i * 8;
                                if (count >= max_out) return count;
                            }
                        }
                    }
                    offset += step;
                }
                break;
            }

            default:
                break;
        }

        // Scalar remainder for items < 16 bytes
        while (offset < limit && count < max_out) {
            int matched = 0;
            switch (data_type) {
                case TYPE_INT32: {
                    int32_t a = *(const int32_t *)(old_buf + offset);
                    int32_t b = *(const int32_t *)(new_buf + offset);
                    if (comp_type == COMP_CHANGED) matched = (a != b);
                    else if (comp_type == COMP_UNCHANGED) matched = (a == b);
                    else if (comp_type == COMP_INCREASED) matched = (b > a);
                    else if (comp_type == COMP_DECREASED) matched = (b < a);
                    else if (comp_type == COMP_EXACT_DIFFERENCE) matched = ((b - a) == (int32_t)target_val);
                    break;
                }
                case TYPE_UINT32: {
                    uint32_t a = *(const uint32_t *)(old_buf + offset);
                    uint32_t b = *(const uint32_t *)(new_buf + offset);
                    if (comp_type == COMP_CHANGED) matched = (a != b);
                    else if (comp_type == COMP_UNCHANGED) matched = (a == b);
                    else if (comp_type == COMP_INCREASED) matched = (b > a);
                    else if (comp_type == COMP_DECREASED) matched = (b < a);
                    else if (comp_type == COMP_EXACT_DIFFERENCE) matched = ((uint32_t)(b - a) == (uint32_t)target_val);
                    break;
                }
                case TYPE_FLOAT: {
                    float a = *(const float *)(old_buf + offset);
                    float b = *(const float *)(new_buf + offset);
                    if (comp_type == COMP_CHANGED) matched = (a != b);
                    else if (comp_type == COMP_UNCHANGED) matched = (a == b);
                    else if (comp_type == COMP_INCREASED) matched = (b > a);
                    else if (comp_type == COMP_DECREASED) matched = (b < a);
                    else if (comp_type == COMP_EXACT_DIFFERENCE) matched = (fabsf((b - a) - (float)target_val) < 1e-4f);
                    break;
                }
                case TYPE_INT64: {
                    int64_t a = *(const int64_t *)(old_buf + offset);
                    int64_t b = *(const int64_t *)(new_buf + offset);
                    if (comp_type == COMP_CHANGED) matched = (a != b);
                    else if (comp_type == COMP_UNCHANGED) matched = (a == b);
                    else if (comp_type == COMP_INCREASED) matched = (b > a);
                    else if (comp_type == COMP_DECREASED) matched = (b < a);
                    else if (comp_type == COMP_EXACT_DIFFERENCE) matched = ((b - a) == (int64_t)target_val);
                    break;
                }
                case TYPE_UINT64: {
                    uint64_t a = *(const uint64_t *)(old_buf + offset);
                    uint64_t b = *(const uint64_t *)(new_buf + offset);
                    if (comp_type == COMP_CHANGED) matched = (a != b);
                    else if (comp_type == COMP_UNCHANGED) matched = (a == b);
                    else if (comp_type == COMP_INCREASED) matched = (b > a);
                    else if (comp_type == COMP_DECREASED) matched = (b < a);
                    else if (comp_type == COMP_EXACT_DIFFERENCE) matched = ((uint64_t)(b - a) == (uint64_t)target_val);
                    break;
                }
                case TYPE_DOUBLE: {
                    double a = *(const double *)(old_buf + offset);
                    double b = *(const double *)(new_buf + offset);
                    if (comp_type == COMP_CHANGED) matched = (a != b);
                    else if (comp_type == COMP_UNCHANGED) matched = (a == b);
                    else if (comp_type == COMP_INCREASED) matched = (b > a);
                    else if (comp_type == COMP_DECREASED) matched = (b < a);
                    else if (comp_type == COMP_EXACT_DIFFERENCE) matched = (fabs((b - a) - target_val) < 1e-4);
                    break;
                }
            }

            if (matched) {
                out_offsets[count++] = offset;
            }
            offset += item_size;
        }
    }

    return count;
}

// -------------------------------------------------------------------------
// Multi-threaded Grand Central Dispatch (GCD) Accelerated Diffing
// -------------------------------------------------------------------------

size_t diff_snapshots_gcd(
    const uint8_t *old_buf,
    const uint8_t *new_buf,
    size_t size,
    int comp_type,
    int data_type,
    double target_val,
    uint64_t *out_offsets,
    size_t max_out
) {
    if (!old_buf || !new_buf || size == 0 || !out_offsets || max_out == 0) {
        return 0;
    }

    // For smaller regions (< 2 MB), GCD dispatch overhead exceeds execution time
    if (size < 2 * 1024 * 1024) {
        return diff_snapshots_neon(old_buf, new_buf, size, comp_type, data_type, target_val, out_offsets, max_out);
    }

    size_t chunk_size = 2 * 1024 * 1024; // 2 MB per slice
    size_t num_chunks = (size + chunk_size - 1) / chunk_size;
    if (num_chunks > 16) num_chunks = 16;
    chunk_size = (size + num_chunks - 1) / num_chunks;
    chunk_size = (chunk_size / 64) * 64; // align to 64 bytes

    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    typedef struct {
        uint64_t *matches;
        size_t count;
    } ChunkResult;

    ChunkResult *results = (ChunkResult *)calloc(num_chunks, sizeof(ChunkResult));
    for (size_t i = 0; i < num_chunks; ++i) {
        results[i].matches = (uint64_t *)malloc(max_out * sizeof(uint64_t));
        results[i].count = 0;
    }

    dispatch_apply(num_chunks, queue, ^(size_t idx) {
        size_t c_start = idx * chunk_size;
        if (c_start >= size) return;
        size_t c_size = (c_start + chunk_size <= size) ? chunk_size : (size - c_start);

        results[idx].count = diff_snapshots_neon(
            old_buf + c_start,
            new_buf + c_start,
            c_size,
            comp_type,
            data_type,
            target_val,
            results[idx].matches,
            max_out
        );
        // Adjust offsets to global buffer relative
        for (size_t m = 0; m < results[idx].count; ++m) {
            results[idx].matches[m] += c_start;
        }
    });

    size_t total_merged = 0;
    for (size_t i = 0; i < num_chunks && total_merged < max_out; ++i) {
        size_t to_copy = results[i].count;
        if (total_merged + to_copy > max_out) {
            to_copy = max_out - total_merged;
        }
        if (to_copy > 0) {
            memcpy(out_offsets + total_merged, results[i].matches, to_copy * sizeof(uint64_t));
            total_merged += to_copy;
        }
    }

    for (size_t i = 0; i < num_chunks; ++i) {
        free(results[i].matches);
    }
    free(results);

    return total_merged;
}
