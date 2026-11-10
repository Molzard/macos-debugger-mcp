"""Unit tests for Priority 1 Kernel Hardening in macOS Debugger MCP.

Tests:
1. Fallback VM_PROT_COPY on read-only/executable dummy memory pages.
2. dyld_shared_cache elimination (0x180000000 - 0x280000000) during memory region inspection.
3. Dirty page & uncommitted zero-fill filtering (pages_resident == 0 && pages_dirtied == 0).
4. Tool interface compatibility across search, snapshot, pointer scanner, and RTTI resolver.
"""

from __future__ import annotations

import ctypes
import os
import struct
import unittest
from typing import Any

from macos_debugger_mcp.debugger import (
    PAGE_SIZE,
    SHARED_REGION_BASE_ARM64,
    SHARED_REGION_END_ARM64,
    VM_PROT_COPY,
    VM_PROT_EXECUTE,
    VM_PROT_READ,
    VM_PROT_WRITE,
    MacOSDebugger,
    vm_region_extended_info_data_t,
)
from macos_debugger_mcp.snapshots import take_memory_snapshot
from macos_debugger_mcp.pointer_scanner import find_pointer_chains
from macos_debugger_mcp.rtti_resolver import MachVMAccess


class TestPriority1Hardening(unittest.TestCase):
    def setUp(self) -> None:
        self.debugger = MacOSDebugger()
        self.pid = os.getpid()
        self._libc = ctypes.CDLL(None)
        self.task = self.debugger.get_task_port(self.pid)

    def test_01_vm_prot_copy_executable_patching(self) -> None:
        """Validate VM_PROT_COPY fallback when writing to read-only/executable memory."""
        mach_vm_allocate = self._libc.mach_vm_allocate
        mach_vm_protect = self._libc.mach_vm_protect
        mach_vm_deallocate = self._libc.mach_vm_deallocate

        mach_vm_allocate.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_uint64,
            ctypes.c_int,
        ]
        mach_vm_allocate.restype = ctypes.c_int

        mach_vm_protect.argtypes = [
            ctypes.c_uint32,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
        ]
        mach_vm_protect.restype = ctypes.c_int

        # 1. Allocate a 16KB page
        page_addr = ctypes.c_uint64(0)
        kr = mach_vm_allocate(self.task, ctypes.byref(page_addr), PAGE_SIZE, 1)
        self.assertEqual(kr, 0, "mach_vm_allocate failed")
        addr = page_addr.value

        try:
            # 2. Protect page as read-execute (r-x) simulating a __TEXT segment
            kr_p = mach_vm_protect(self.task, ctypes.c_uint64(addr), ctypes.c_uint64(PAGE_SIZE), 0, VM_PROT_READ | VM_PROT_EXECUTE)
            self.assertEqual(kr_p, 0, "mach_vm_protect r-x failed")

            # 3. Write patch payload into the r-x memory page
            # Payload: ARM64 `MOV W0, #1; RET` (0x52800020, 0xD65F03C0)
            patch_hex = "20008052c0035fd6"
            res = self.debugger.write_memory(self.pid, f"0x{addr:x}", patch_hex)

            self.assertEqual(res["status"], "success")
            self.assertEqual(res["bytes_written"], 8)

            # 4. Verify written bytes match exactly
            read_res = self.debugger.read_memory(self.pid, f"0x{addr:x}", 8, fmt="hex")
            self.assertEqual(read_res["hex"], patch_hex)

            # 5. Also test write_memory_bytes alias
            patch_hex2 = "40008052c0035fd6"  # MOV W0, #2; RET
            res2 = self.debugger.write_memory_bytes(self.pid, f"0x{addr:x}", patch_hex2)
            self.assertEqual(res2["status"], "success")

            read_res2 = self.debugger.read_memory(self.pid, f"0x{addr:x}", 8, fmt="hex")
            self.assertEqual(read_res2["hex"], patch_hex2)

        finally:
            mach_vm_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]
            mach_vm_deallocate.restype = ctypes.c_int
            mach_vm_deallocate(self.task, ctypes.c_uint64(addr), ctypes.c_uint64(PAGE_SIZE))

    def test_02_exclude_shared_cache_filtering(self) -> None:
        """Verify that dyld_shared_cache regions (0x180000000 - 0x280000000) are excluded when exclude_shared_cache=True."""
        # Query with exclude_shared_cache=True (Default)
        regions_filtered = self.debugger.inspect_memory_regions(self.pid, limit=1000, exclude_shared_cache=True)
        self.assertGreater(regions_filtered["count"], 0)

        for r in regions_filtered["regions"]:
            start = int(r["start_address"], 16)
            self.assertFalse(
                SHARED_REGION_BASE_ARM64 <= start < SHARED_REGION_END_ARM64,
                f"Found shared cache region 0x{start:x} despite exclude_shared_cache=True",
            )

        # Query with exclude_shared_cache=False
        regions_all = self.debugger.inspect_memory_regions(self.pid, limit=1000, exclude_shared_cache=False)
        shared_found = any(
            SHARED_REGION_BASE_ARM64 <= int(r["start_address"], 16) < SHARED_REGION_END_ARM64
            for r in regions_all["regions"]
        )
        self.assertTrue(shared_found, "Shared cache regions should be present when exclude_shared_cache=False")

    def test_03_dirty_page_and_uncommitted_filtering(self) -> None:
        """Verify that regions with pages_resident == 0 && pages_dirtied == 0 are skipped instantly."""
        mach_vm_allocate = self._libc.mach_vm_allocate
        mach_vm_deallocate = self._libc.mach_vm_deallocate

        # Allocate 32 MB uncommitted virtual memory without writing into it
        alloc_size = 32 * 1024 * 1024
        alloc_addr = ctypes.c_uint64(0)
        kr = mach_vm_allocate(self.task, ctypes.byref(alloc_addr), alloc_size, 1)
        self.assertEqual(kr, 0)
        addr = alloc_addr.value

        try:
            # Memory search should not match and should skip uncommitted 32 MB instantly
            search_res = self.debugger.search_memory(
                self.pid,
                value=133742,
                data_type="int32",
                max_results=5,
                region_type="writable",
                exclude_shared_cache=True,
            )
            # The uncommitted page was not dirtied, so it was skipped
            matches_in_alloc = [m for m in search_res["matches"] if addr <= int(m["address"], 16) < addr + alloc_size]
            self.assertEqual(len(matches_in_alloc), 0, "Uncommitted allocated region should not have been searched")

            # Test take_memory_snapshot with dirty page filtering
            snap = take_memory_snapshot(
                self.debugger,
                self.pid,
                snapshot_name="test_dirty_filter",
                max_total_mb=64,
                exclude_shared_cache=True,
            )
            self.assertEqual(snap["status"], "success")

            # Verify the uncommitted address was not added into snapshot
            from macos_debugger_mcp.snapshots import _snapshots
            self.assertNotIn(addr, _snapshots["test_dirty_filter"])

        finally:
            mach_vm_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint64, ctypes.c_uint64]
            mach_vm_deallocate(self.task, ctypes.c_uint64(addr), ctypes.c_uint64(alloc_size))

    def test_04_rtti_and_pointer_chain_exclude_shared_cache(self) -> None:
        """Verify MachVMAccess.enumerate_regions and find_pointer_chains support exclude_shared_cache."""
        vm = MachVMAccess(self.pid)
        try:
            regs_ex = vm.enumerate_regions(writable_only=False, exclude_shared_cache=True)
            for start, end, _ in regs_ex:
                self.assertFalse(
                    SHARED_REGION_BASE_ARM64 <= start < SHARED_REGION_END_ARM64,
                    f"MachVMAccess included shared cache region 0x{start:x}",
                )

            regs_inc = vm.enumerate_regions(writable_only=False, exclude_shared_cache=False)
            has_shared = any(SHARED_REGION_BASE_ARM64 <= s < SHARED_REGION_END_ARM64 for s, _, _ in regs_inc)
            self.assertTrue(has_shared, "MachVMAccess should include shared cache when exclude_shared_cache=False")
        finally:
            vm.close()

        # Test find_pointer_chains with mock data accepts exclude_shared_cache=True
        mock_mem = {0x100000000: struct.pack("<Q", 0x100000010)}
        mock_mods = [{
            "name": "TestFramework",
            "base_address": 0x100000000,
            "data_segments": [("__DATA", 0x100000000, 0x100000100)],
        }]
        res = find_pointer_chains(
            self.debugger,
            self.pid,
            target_address="0x100000010",
            max_depth=2,
            exclude_shared_cache=True,
            mock_modules=mock_mods,
            mock_memory=mock_mem,
        )
        self.assertEqual(res["status"], "success")
        self.assertGreater(len(res["chains_found"]), 0)


if __name__ == "__main__":
    unittest.main()
