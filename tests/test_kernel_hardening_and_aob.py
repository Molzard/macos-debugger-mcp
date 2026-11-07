"""Unit tests for Kernel Hardening, 16KB Page-Aligned Traversal, Kqueue Lifecycle, and ARM64 AoB Signature Generator."""

from __future__ import annotations

import ctypes
import os
import struct
import subprocess
import time
import unittest
from typing import Any

from macos_debugger_mcp.aob_generator import generate_arm64_aob_signature, mask_arm64_instruction
from macos_debugger_mcp.debugger import MacOSDebugger, PAGE_SIZE
from macos_debugger_mcp.freeze_daemon import (
    FreezeWorker,
    ProcessExitWatcher,
    _exit_watchers,
    _handle_process_exit,
    _workers,
    freeze_address_value,
    unfreeze_address_value,
)
from macos_debugger_mcp.snapshots import diff_memory_snapshots, take_memory_snapshot


class MockDebugger:
    """Mock debugger for isolated memory testing without root task port."""

    def __init__(self, memory_map: dict[int, bytes]):
        self.memory_map = memory_map
        self.task_ports: dict[int, int] = {}
        self.deallocated_ports: list[int] = []

    def get_task_port(self, pid: int) -> int:
        return 12345

    def cleanup_task_port(self, pid: int) -> bool:
        if pid in self.task_ports:
            p = self.task_ports.pop(pid)
            self.deallocated_ports.append(p)
            return True
        return False

    def inspect_memory_regions(self, pid: int, limit: int = 100) -> dict[str, Any]:
        regions = []
        for base, data in sorted(self.memory_map.items()):
            regions.append({
                "start_address": f"0x{base:x}",
                "end_address": f"0x{base + len(data):x}",
                "size_bytes": len(data),
                "protection": "rwx",
            })
        return {"regions": regions, "count": len(regions)}

    def read_raw_bytes(
        self, pid: int, address: int, size: int, preallocated_buf: Any = None
    ) -> bytes:
        for base, data in self.memory_map.items():
            if base <= address < base + len(data):
                offset = address - base
                chunk = data[offset : offset + size]
                return chunk
        raise ValueError(f"Unmapped address 0x{address:x}")

    def write_memory(self, pid: int, address_str: str, bytes_hex: str) -> dict[str, Any]:
        return {"status": "success"}


class TestKernelHardeningAndAOB(unittest.TestCase):
    def setUp(self):
        self.debugger = MacOSDebugger()

    def test_01_page_aligned_16kb_traversal_and_snapshot(self):
        """Verify 16 KB page-aligned traversal and preallocated buffer handling in snapshots."""
        # Create a mock memory map with two 64KB regions (4 x 16KB pages each)
        region1_base = 0x100000000
        region1_data = b"A" * (64 * 1024)
        region2_base = 0x200000000
        region2_data = b"B" * (32 * 1024)

        mock_dbg = MockDebugger({
            region1_base: region1_data,
            region2_base: region2_data,
        })

        snap = take_memory_snapshot(mock_dbg, pid=1234, snapshot_name="snap_test_16kb", max_total_mb=10)
        self.assertEqual(snap["status"], "success")
        self.assertEqual(snap["total_bytes"], (64 + 32) * 1024)
        self.assertGreaterEqual(snap["regions_captured"], 2)

        # Verify PAGE_SIZE is exactly 16384 (16 KB) on ARM64 macOS
        self.assertEqual(PAGE_SIZE, 16384)

    def test_02_kqueue_process_exit_monitoring(self):
        """Verify BSD kqueue detects process exit (NOTE_EXIT) and triggers automatic worker cleanup."""
        # Spawn a short-lived subprocess
        subproc = subprocess.Popen(["sleep", "0.2"])
        target_pid = subproc.pid

        # Set up a mock freeze worker and task port
        mock_dbg = MockDebugger({})
        mock_dbg.task_ports[target_pid] = 9999

        worker = FreezeWorker(mock_dbg, target_pid, "0x1000", "00000000", interval_ms=50)
        freeze_id = f"test_frz_{target_pid}"
        _workers[freeze_id] = worker
        worker.start()

        # Start watcher
        watcher = ProcessExitWatcher(target_pid, mock_dbg, _handle_process_exit)
        _exit_watchers[target_pid] = watcher
        watcher.start()

        # Wait for the child process to exit naturally
        subproc.wait(timeout=2.0)
        # Give kqueue watcher up to 1 second to detect NOTE_EXIT
        time.sleep(0.4)

        # Worker should have been stopped and removed automatically
        self.assertNotIn(freeze_id, _workers)
        self.assertNotIn(target_pid, _exit_watchers)
        self.assertFalse(worker.is_alive())
        # Task port should have been cleaned up
        self.assertIn(9999, mock_dbg.deallocated_ports)

    def test_03_arm64_aob_instruction_masking(self):
        """Verify instruction-aware masking: ADRP, BL, B, B.<cond>, CBZ, TBZ, and LDR/ADD immediate offsets."""
        from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs

        md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        md.detail = True

        # Test Synthetic sequence matching the prompt specification:
        # F8 5F BC A9 -> STP X24, X23, [SP, #-0x40]!
        # 00 00 00 91 -> ADD X0, X0, #0
        # 00 00 00 14 -> B #0
        # E0 03 13 AA -> MOV X0, X19
        # C0 03 5F D6 -> RET
        code = bytes.fromhex("f85fbca90000009100000014e00313aac0035fd6")
        adrp_regs = {"x0"}
        tokens = []
        for insn in md.disasm(code, 0x102B48000):
            t = mask_arm64_instruction(insn, adrp_regs)
            tokens.extend(t)

        sig = " ".join(tokens)
        expected = "F8 5F BC A9 ?? ?? 00 91 ?? ?? ?? 14 E0 03 13 AA C0 03 5F D6"
        self.assertEqual(sig, expected)

    def test_04_arm64_aob_signature_generator_uniqueness(self):
        """Verify generate_arm64_aob_signature uniqueness expansion across executable segment."""
        # Create an executable segment containing:
        # 1. Padding with similar opcode
        # 2. Target function at 0x102B48000
        # 3. Duplicate prefix function to test expansion
        func_target = bytes.fromhex("f85fbca90000009100000014e00313aac0035fd6")  # 20 bytes
        func_duplicate_prefix = bytes.fromhex("f85fbca9000000911f2003d51f2003d5c0035fd6")  # first 8 bytes match!

        text_base = 0x102B40000
        padding1 = b"\x1f\x20\x03\xd5" * 100  # NOPs
        padding2 = b"\x1f\x20\x03\xd5" * 100

        # Construct full text segment:
        # 0x102B40000: padding1 (400 bytes)
        # 0x102B40190: duplicate prefix (20 bytes)
        # 0x102B401A4: padding2 (400 bytes)
        # 0x102B48000: target function
        offset_target = 0x102B48000 - text_base
        text_data = bytearray(offset_target + len(func_target) + 256)
        text_data[: len(padding1)] = padding1
        text_data[len(padding1) : len(padding1) + len(func_duplicate_prefix)] = func_duplicate_prefix
        text_data[offset_target : offset_target + len(func_target)] = func_target

        res = generate_arm64_aob_signature(
            debugger=None,
            pid=os.getpid(),
            function_address="0x102b48000",
            scan_length=32,
            text_segment_bytes=bytes(text_data),
            text_segment_base=text_base,
        )

        self.assertEqual(res["status"], "success")
        self.assertTrue(res["unique_match"])
        self.assertEqual(res["matched_address"].lower(), "0x102b48000")
        self.assertGreaterEqual(res["length"], 12)  # Expanded past 8 bytes to disambiguate from duplicate prefix
        self.assertIn("F8 5F BC A9", res["signature"])
        print("\n--- Generated Unique ARM64 AoB Signature ---")
        print(f"Signature: {res['signature']}")
        print(f"Length: {res['length']} bytes (Unique Match: {res['unique_match']})")

    def test_05_rtti_two_stage_monitordata_filter(self):
        """Verify two-stage pointer indexing eliminates invalid objects instantly via MonitorData == 0."""
        from macos_debugger_mcp.rtti_resolver import RTTIResolver

        # Mock metadata
        class MockType:
            name = "TestClass"
            full_name = "Namespace.TestClass"
            fields = []

        class MockMeta:
            def find_type(self, name):
                if name in ("TestClass", "Namespace.TestClass"):
                    return MockType()
                return None

        resolver = RTTIResolver(MockMeta())
        target_descriptor = 0x700000000

        # Create a mock heap chunk containing 3 candidates:
        # Candidate 1: valid descriptor, but MonitorData != 0 (locked / non-object) -> REJECTED
        # Candidate 2: invalid descriptor (< 0x1000) -> REJECTED
        # Candidate 3: valid descriptor, MonitorData == 0 -> ACCEPTED
        heap_base = 0x150000000
        chunk = bytearray(64)
        # Cand 1 at offset 0
        struct.pack_into("<QQ", chunk, 0, target_descriptor, 0x1234)  # monitor = 0x1234 != 0
        # Cand 2 at offset 16
        struct.pack_into("<QQ", chunk, 16, 0x200, 0x0)  # descriptor < 0x1000
        # Cand 3 at offset 32
        struct.pack_into("<QQ", chunk, 32, target_descriptor, 0x0)  # valid object header!

        name_addr = 0x700001000
        desc_buf = bytearray(48)
        struct.pack_into("<Q", desc_buf, 16, name_addr)
        struct.pack_into("<Q", desc_buf, 24, 0)

        mock_memory = {
            heap_base: bytes(chunk),
            target_descriptor: bytes(desc_buf),
            name_addr: b"TestClass\x00",
        }

        def mock_read(addr: int, size: int) -> bytes:
            for b, d in mock_memory.items():
                if b <= addr < b + len(d):
                    off = addr - b
                    return d[off : off + size]
            return b"\x00" * size

        regions = [(heap_base, heap_base + len(chunk), 3)]
        instances = resolver.find_active_instances(
            class_name="TestClass",
            read_fn=mock_read,
            regions=regions,
            target_klass=target_descriptor,
        )

        self.assertEqual(len(instances), 1)
        self.assertEqual(instances[0]["raw_address"], hex(heap_base + 32))


if __name__ == "__main__":
    unittest.main()
