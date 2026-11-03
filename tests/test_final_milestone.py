"""Unit and Integration Tests for Final Milestone: Snapshots, Freezing, and Dylib Exporter."""

import ctypes
import os
import struct
import sys
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from macos_debugger_mcp.debugger import MacOSDebugger
from macos_debugger_mcp.dylib_exporter import export_dylib_patch
from macos_debugger_mcp.freeze_daemon import freeze_address_value, unfreeze_address_value
from macos_debugger_mcp.snapshots import _snapshots, diff_memory_snapshots, take_memory_snapshot


class TestFinalMilestone(unittest.TestCase):
    def setUp(self):
        self.debugger = MacOSDebugger()

    def test_01_snapshot_and_diff(self):
        """Verify memory snapshotting and differential mutation filtering."""
        # Create synthetic snapshots
        base_addr = 0x100000000
        # 16 bytes: 4 x int32
        # val0: 100 -> 100 (unchanged)
        # val1: 200 -> 300 (increased by 100)
        # val2: 50 -> 49   (decreased by 1)
        # val3: 10 -> 20   (changed)
        data_a = struct.pack("<4i", 100, 200, 50, 10)
        data_b = struct.pack("<4i", 100, 300, 49, 20)

        _snapshots["snap_initial"] = {base_addr: data_a}
        _snapshots["snap_subsequent"] = {base_addr: data_b}

        # Filter: changed
        diff_changed = diff_memory_snapshots("snap_initial", "snap_subsequent", data_type="int32", filter_type="changed")
        self.assertEqual(diff_changed["status"], "success")
        self.assertEqual(diff_changed["total_matches"], 3)

        # Filter: unchanged
        diff_unchanged = diff_memory_snapshots("snap_initial", "snap_subsequent", data_type="int32", filter_type="unchanged")
        self.assertEqual(diff_unchanged["total_matches"], 1)
        self.assertEqual(diff_unchanged["matches"][0]["value_before"], 100)

        # Filter: decreased
        diff_decreased = diff_memory_snapshots("snap_initial", "snap_subsequent", data_type="int32", filter_type="decreased")
        self.assertEqual(diff_decreased["total_matches"], 1)
        self.assertEqual(diff_decreased["matches"][0]["diff"], -1)

        # Filter: exact_difference = 100
        diff_exact = diff_memory_snapshots(
            "snap_initial", "snap_subsequent", data_type="int32", filter_type="exact_difference", diff_value=100
        )
        self.assertEqual(diff_exact["total_matches"], 1)
        self.assertEqual(diff_exact["matches"][0]["value_after"], 300)

    def test_02_memory_freeze_worker(self):
        """Verify background thread asserts memory value repeatedly."""
        # Allocate writable variable in self process
        target_var = ctypes.c_int32(42)
        addr = ctypes.addressof(target_var)
        addr_hex = f"0x{addr:x}"

        # Target payload: Int32 = 999 (0x000003e7 -> 'e7030000')
        payload_hex = "e7030000"

        # Freeze address
        res = freeze_address_value(self.debugger, os.getpid(), addr_hex, payload_hex, interval_ms=20)
        self.assertEqual(res["status"], "frozen")
        freeze_id = res["freeze_id"]

        # Initial write should have applied
        self.assertEqual(target_var.value, 999)

        # External code attempts to mutate the value
        target_var.value = 123
        self.assertEqual(target_var.value, 123)

        # Wait for daemon assertion tick
        time.sleep(0.08)
        self.assertEqual(target_var.value, 999)

        # Unfreeze
        unfrz = unfreeze_address_value(freeze_id)
        self.assertEqual(unfrz["status"], "unfrozen")
        self.assertGreater(unfrz["total_writes"], 1)

        # External mutation should now persist
        target_var.value = 555
        time.sleep(0.05)
        self.assertEqual(target_var.value, 555)

    def test_03_dylib_patch_export_and_compile(self):
        """Verify standalone ARM64 dylib patch generation and clang compilation."""
        dist_dir = "/Users/alfaroo/Sintax/MCP/macos-debugger-mcp/dist_patches"
        patch_def = [
            {
                "offset": "0x1234",
                "patch_bytes": "200080d2c0035fd6",  # MOV X0, #1; RET
                "description": "Mock Bird Return Hook",
            },
            {
                "offset": "0x5678",
                "patch_bytes": "1f2003d5",  # NOP
                "description": "Mock Score Check Bypass",
            },
        ]

        res = export_dylib_patch(
            patch_name="TestInfiniteBirdsPatch",
            target_framework="UnityFramework",
            patches=patch_def,
            output_dir=dist_dir,
        )

        self.assertEqual(res["status"], "success")
        self.assertTrue(os.path.isfile(res["dylib_path"]))
        self.assertTrue(os.path.isfile(res["source_c_path"]))
        self.assertTrue(os.path.isfile(res["makefile_path"]))
        self.assertGreater(os.path.getsize(res["dylib_path"]), 0)

        # Verify Mach-O ARM64 header of compiled dylib
        with open(res["dylib_path"], "rb") as f:
            magic = struct.unpack("<I", f.read(4))[0]
            # MH_MAGIC_64 = 0xfeedfacf
            self.assertEqual(magic, 0xFEEDFACF)

        print("\n--- Dylib Compilation Success ---")
        print(f"Path: {res['dylib_path']} ({os.path.getsize(res['dylib_path'])} bytes)")
        print(f"Source: {res['source_c_path']}")


if __name__ == "__main__":
    unittest.main()
