"""Unit tests for Priority 2 Native C/SIMD Acceleration & Zero-Copy CoW Snapshot."""

import ctypes
import os
import platform
import random
import struct
import unittest
from typing import List

from macos_debugger_mcp.accel import (
    aob_scan_all_neon,
    aob_scan_neon,
    diff_snapshots_neon,
    load_accel_core,
)
from macos_debugger_mcp.snapshots import (
    Snapshot,
    _mach_task_self,
    _mach_vm_deallocate,
    _mach_vm_remap,
    _snapshots,
    clear_all_snapshots,
    diff_memory_snapshots,
    release_memory_snapshot,
    take_memory_snapshot,
)


class MockCoWDebugger:
    """Mock debugger providing task port for mach_vm_remap testing."""

    def __init__(self, regions_dict=None):
        self.regions_dict = regions_dict or {}

    def get_task_port(self, pid: int) -> int:
        return _mach_task_self()

    def inspect_memory_regions(self, pid: int, limit: int = 10000, exclude_shared_cache: bool = True):
        regions = []
        for base, data in self.regions_dict.items():
            regions.append({
                "start_address": hex(base),
                "size_bytes": len(data),
                "protection": "rw-",
                "share_mode": 0,
                "user_tag": "VM_MEMORY_MALLOC",
                "pages_resident": 1,
                "pages_dirtied": 1,
            })
        return {"regions": regions}

    def read_raw_bytes(self, pid: int, address: int, size: int, preallocated_buf=None) -> bytes:
        for base, data in self.regions_dict.items():
            if base <= address < base + len(data):
                offset = address - base
                return bytes(data[offset : offset + size])
        return b"\x00" * size


class TestPriority2Acceleration(unittest.TestCase):
    """Priority 2 verification suite for ARM64 NEON, GCD, and Zero-Copy CoW."""

    def setUp(self):
        clear_all_snapshots()

    def tearDown(self):
        clear_all_snapshots()

    def test_01_accel_core_loader(self):
        """Verify libmcp_accel.dylib compiles and loads with proper symbols."""
        if platform.machine() != "arm64" or platform.system() != "Darwin":
            self.skipTest("Requires macOS Apple Silicon (ARM64)")

        lib = load_accel_core()
        self.assertIsNotNone(lib, "libmcp_accel.dylib failed to load or compile")
        self.assertTrue(hasattr(lib, "aob_scan_neon"))
        self.assertTrue(hasattr(lib, "aob_scan_all_neon"))
        self.assertTrue(hasattr(lib, "diff_snapshots_neon"))
        self.assertTrue(hasattr(lib, "diff_snapshots_gcd"))

    def test_02_wildcard_aob_scan_neon(self):
        """Verify high-speed wildcard pattern scanning with ARM64 NEON vector reduction."""
        # Create 2 MB buffer with targeted signatures at specific offsets
        data = bytearray(b"\xaa" * (2 * 1024 * 1024))

        # Insert signature 1: F8 5F ?? A9 00 00 ?? 91 (at offset 12345)
        sig1 = b"\xf8\x5f\xbc\xa9\x00\x00\x42\x91"
        data[12345 : 12345 + 8] = sig1

        # Insert signature 2: F8 5F ?? A9 00 00 ?? 91 (at offset 1000000)
        sig2 = b"\xf8\x5f\x12\xa9\x00\x00\x99\x91"
        data[1000000 : 1000000 + 8] = sig2

        pattern = b"\xf8\x5f\x00\xa9\x00\x00\x00\x91"
        mask =    b"\xff\xff\x00\xff\xff\xff\x00\xff"

        # 1. Single scan (first match)
        first_idx = aob_scan_neon(bytes(data), len(data), pattern, mask)
        self.assertEqual(first_idx, 12345, f"Expected offset 12345, got {first_idx}")

        # 2. All scan (multi-match)
        all_matches = aob_scan_all_neon(bytes(data), len(data), pattern, mask, max_results=10)
        self.assertEqual(len(all_matches), 2)
        self.assertEqual(all_matches[0], 12345)
        self.assertEqual(all_matches[1], 1000000)

        # 3. Non-matching pattern
        no_match = aob_scan_neon(bytes(data), len(data), b"\xde\xad\xbe\xef", b"\xff\xff\xff\xff")
        self.assertEqual(no_match, -1)

    def test_03_simd_diff_vs_naive_python(self):
        """Validate SIMD NEON diffing results match exact naive Python comparison."""
        size = 128 * 1024  # 128 KB
        buf_old = bytearray(b"\x00" * size)
        buf_new = bytearray(b"\x00" * size)

        # Populate with structured test values
        random.seed(42)
        changed_indices = [128, 512, 1024, 4096, 16384, 32768, 65536, 100000]

        for idx in changed_indices:
            v_old = random.randint(100, 500)
            v_new = v_old + 50  # increased by 50
            struct.pack_into("<i", buf_old, idx, v_old)
            struct.pack_into("<i", buf_new, idx, v_new)

        # Test filter: increased
        neon_inc = diff_snapshots_neon(bytes(buf_old), bytes(buf_new), size, comp_type="increased", data_type="int32", max_results=50)
        self.assertEqual(sorted(neon_inc), sorted(changed_indices))

        # Test filter: exact_difference = 50
        neon_diff50 = diff_snapshots_neon(bytes(buf_old), bytes(buf_new), size, comp_type="exact_difference", data_type="int32", target_val=50.0, max_results=50)
        self.assertEqual(sorted(neon_diff50), sorted(changed_indices))

        # Test filter: decreased (should be 0)
        neon_dec = diff_snapshots_neon(bytes(buf_old), bytes(buf_new), size, comp_type="decreased", data_type="int32", max_results=50)
        self.assertEqual(len(neon_dec), 0)

        # Test filter: float32
        struct.pack_into("<f", buf_old, 2048, 12.5)
        struct.pack_into("<f", buf_new, 2048, 37.5) # diff = 25.0
        neon_f32 = diff_snapshots_neon(bytes(buf_old), bytes(buf_new), size, comp_type="exact_difference", data_type="float", target_val=25.0, max_results=10)
        self.assertIn(2048, neon_f32)

    def test_04_fast_page_skip_16kb(self):
        """Verify 16KB fast page-skip correctly bypasses untouched memory blocks."""
        page_size = 16384
        total_size = page_size * 10  # 160 KB (10 pages)

        buf_old = bytearray(b"\x55" * total_size)
        buf_new = bytearray(b"\x55" * total_size)

        # Modify only page 3 and page 8
        struct.pack_into("<i", buf_new, page_size * 3 + 100, 9999)
        struct.pack_into("<i", buf_new, page_size * 8 + 400, 8888)

        matches = diff_snapshots_neon(bytes(buf_old), bytes(buf_new), total_size, comp_type="changed", data_type="int32", max_results=10)
        self.assertEqual(len(matches), 2)
        self.assertIn(page_size * 3 + 100, matches)
        self.assertIn(page_size * 8 + 400, matches)

    def test_05_mach_vm_remap_zero_copy_cow(self):
        """Verify mach_vm_remap(copy=True) creates true CoW zero-copy page table mapping."""
        libc = ctypes.CDLL(None)
        mach_task_self = libc.mach_task_self
        mach_vm_allocate = libc.mach_vm_allocate

        page_size = 16384
        src_addr = ctypes.c_uint64(0)
        kr = mach_vm_allocate(mach_task_self(), ctypes.byref(src_addr), page_size, 1)  # VM_FLAGS_ANYWHERE
        self.assertEqual(kr, 0, "mach_vm_allocate failed")

        try:
            # Write initial signature into page
            initial_sig = b"TEST_SIGNATURE_ZERO_COPY_ARM64_COW_12345"
            ctypes.memmove(src_addr.value + 64, initial_sig, len(initial_sig))

            # Take memory snapshot using CoW
            snapshot = Snapshot("cow_test_snap", os.getpid())
            local_addr = ctypes.c_uint64(0)
            cur_prot = ctypes.c_int32(0)
            max_prot = ctypes.c_int32(0)

            self.assertIsNotNone(_mach_vm_remap, "mach_vm_remap symbol missing")
            assert _mach_vm_remap is not None
            kr_remap = _mach_vm_remap(
                mach_task_self(),
                ctypes.byref(local_addr),
                ctypes.c_uint64(page_size),
                0,
                1,  # VM_FLAGS_ANYWHERE
                mach_task_self(),
                src_addr.value,
                True,  # copy = True (CoW)
                ctypes.byref(cur_prot),
                ctypes.byref(max_prot),
                2,  # VM_INHERIT_NONE
            )
            self.assertEqual(kr_remap, 0, f"mach_vm_remap failed: kr={kr_remap}")

            snapshot.add_remapped_region(src_addr.value, local_addr.value, page_size, local_addr.value, page_size)
            _snapshots["cow_test_snap"] = snapshot

            # Verify snapshot contains exact memory
            snap_buf = snapshot[src_addr.value]
            self.assertEqual(bytes(snap_buf[64 : 64 + len(initial_sig)]), initial_sig)

            # Mutate original memory!
            mutated_sig = b"MUTATED_NEW_VALUE_AFTER_SNAPSHOT_TAKEN!!"
            ctypes.memmove(src_addr.value + 64, mutated_sig, len(mutated_sig))

            # Verify original changed, but snapshot preserved initial value (True CoW!)
            current_live = bytes((ctypes.c_char * len(mutated_sig)).from_address(src_addr.value + 64))
            self.assertEqual(current_live, mutated_sig)

            preserved_snap = bytes(snapshot[src_addr.value][64 : 64 + len(initial_sig)])
            self.assertEqual(preserved_snap, initial_sig, "CoW failed: snapshot reflected live mutation")

        finally:
            if _mach_vm_deallocate:
                _mach_vm_deallocate(mach_task_self(), src_addr, page_size)
            release_memory_snapshot("cow_test_snap")

    def test_06_snapshot_deallocation(self):
        """Verify release_memory_snapshot properly deallocates CoW memory regions."""
        libc = ctypes.CDLL(None)
        mach_task_self = libc.mach_task_self
        mach_vm_allocate = libc.mach_vm_allocate

        page_size = 16384
        src_addr = ctypes.c_uint64(0)
        mach_vm_allocate(mach_task_self(), ctypes.byref(src_addr), page_size, 1)

        try:
            snapshot = Snapshot("cleanup_test", os.getpid())
            local_addr = ctypes.c_uint64(0)
            cur_prot = ctypes.c_int32(0)
            max_prot = ctypes.c_int32(0)

            assert _mach_vm_remap is not None
            _mach_vm_remap(
                mach_task_self(),
                ctypes.byref(local_addr),
                ctypes.c_uint64(page_size),
                0,
                1,
                mach_task_self(),
                src_addr.value,
                True,
                ctypes.byref(cur_prot),
                ctypes.byref(max_prot),
                2,
            )
            snapshot.add_remapped_region(src_addr.value, local_addr.value, page_size, local_addr.value, page_size)
            _snapshots["cleanup_test"] = snapshot

            self.assertIn("cleanup_test", _snapshots)
            self.assertEqual(len(snapshot.remapped_allocations), 1)

            # Release snapshot
            res = release_memory_snapshot("cleanup_test")
            self.assertTrue(res)
            self.assertNotIn("cleanup_test", _snapshots)
            self.assertEqual(len(snapshot.remapped_allocations), 0)

        finally:
            if _mach_vm_deallocate:
                _mach_vm_deallocate(mach_task_self(), src_addr, page_size)

    def test_07_gcd_large_diff_multithreaded(self):
        """Verify GCD parallel diffing handles large multi-megabyte heap buffers."""
        size = 4 * 1024 * 1024  # 4 MB
        buf_a = bytearray(size)
        buf_b = bytearray(size)

        # Plant changes across different 2 MB chunk boundaries
        struct.pack_into("<i", buf_b, 500000, 100)
        struct.pack_into("<i", buf_b, 3000000, 200)

        matches = diff_snapshots_neon(bytes(buf_a), bytes(buf_b), size, comp_type="changed", data_type="int32", use_gcd=True)
        self.assertEqual(len(matches), 2)
        self.assertIn(500000, matches)
        self.assertIn(3000000, matches)


if __name__ == "__main__":
    unittest.main()
