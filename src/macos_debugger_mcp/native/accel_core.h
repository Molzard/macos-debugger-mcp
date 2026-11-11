#ifndef ACCEL_CORE_H
#define ACCEL_CORE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Comparison types
#define COMP_CHANGED          0
#define COMP_UNCHANGED        1
#define COMP_INCREASED        2
#define COMP_DECREASED        3
#define COMP_EXACT_DIFFERENCE 4

// Data types
#define TYPE_INT32   0
#define TYPE_UINT32  1
#define TYPE_INT64   2
#define TYPE_UINT64  3
#define TYPE_FLOAT   4
#define TYPE_DOUBLE  5

/**
 * Wildcard AoB Pattern Scanner accelerated with ARM64 NEON SIMD.
 *
 * @param data Target memory buffer to scan.
 * @param data_len Length of data buffer in bytes.
 * @param pattern Byte pattern array.
 * @param mask Bitmask array (0xFF = exact match, 0x00 = wildcard '??').
 * @param pattern_len Number of bytes in pattern/mask.
 * @return Byte offset of first match, or -1 if not found.
 */
int64_t aob_scan_neon(
    const uint8_t *data,
    size_t data_len,
    const uint8_t *pattern,
    const uint8_t *mask,
    size_t pattern_len
);

/**
 * Find all matching occurrences of wildcard AoB pattern up to max_out.
 */
size_t aob_scan_all_neon(
    const uint8_t *data,
    size_t data_len,
    const uint8_t *pattern,
    const uint8_t *mask,
    size_t pattern_len,
    uint64_t *out_offsets,
    size_t max_out
);

/**
 * Vectorized Snapshot Differencing accelerated with ARM64 NEON & Fast Page-Skip.
 *
 * @param old_buf Memory buffer from snapshot A.
 * @param new_buf Memory buffer from snapshot B.
 * @param size Number of bytes to compare.
 * @param comp_type Comparison mode (COMP_CHANGED, COMP_INCREASED, etc.).
 * @param data_type Element type (TYPE_INT32, TYPE_FLOAT, etc.).
 * @param target_val Target difference value for COMP_EXACT_DIFFERENCE.
 * @param out_offsets Array to write matching byte offsets into.
 * @param max_out Maximum number of matches to record.
 * @return Total number of matching elements recorded in out_offsets.
 */
size_t diff_snapshots_neon(
    const uint8_t *old_buf,
    const uint8_t *new_buf,
    size_t size,
    int comp_type,
    int data_type,
    double target_val,
    uint64_t *out_offsets,
    size_t max_out
);

/**
 * Multi-threaded snapshot diffing via Grand Central Dispatch (GCD).
 */
size_t diff_snapshots_gcd(
    const uint8_t *old_buf,
    const uint8_t *new_buf,
    size_t size,
    int comp_type,
    int data_type,
    double target_val,
    uint64_t *out_offsets,
    size_t max_out
);

#ifdef __cplusplus
}
#endif

#endif // ACCEL_CORE_H
