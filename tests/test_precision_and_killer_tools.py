"""Unit tests for PAC Normalization, Precision AoB Masking, find_pointer_chains, and hook_and_log_method."""

from __future__ import annotations

import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from macos_debugger_mcp.aob_generator import mask_arm64_instruction
from macos_debugger_mcp.debugger import MacOSDebugger
from macos_debugger_mcp.interceptor import hook_and_log_method
from macos_debugger_mcp.pointer_scanner import find_pointer_chains
from macos_debugger_mcp.utils import strip_pac


class TestPrecisionAndKillerTools(unittest.TestCase):
    def setUp(self):
        self.debugger = MacOSDebugger()

    def test_01_strip_pac_normalization(self):
        """Verify PAC / TBI stripping retains clean 48-bit user-space address."""
        clean_addr = 0x104B28190

        # Synthetic PAC tags in bits 48-63
        tagged_ptr1 = clean_addr | (0xE400 << 48)  # 0xe400000104b28190
        tagged_ptr2 = clean_addr | (0x8054 << 48)  # 0x8054000104b28190
        tagged_ptr3 = clean_addr | (0xFFFF << 48)  # Top byte ignore

        self.assertEqual(strip_pac(clean_addr), clean_addr)
        self.assertEqual(strip_pac(tagged_ptr1), clean_addr)
        self.assertEqual(strip_pac(tagged_ptr2), clean_addr)
        self.assertEqual(strip_pac(tagged_ptr3), clean_addr)
        self.assertEqual(strip_pac(None), 0)
        self.assertEqual(strip_pac("invalid"), 0)

    def test_02_precision_aob_masking_bcond_and_ldr_add(self):
        """Verify precision instruction masking for B.<cond> (condition preserved) and LDR/ADD (reg & opcode preserved)."""
        from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs

        md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        md.detail = True

        # 1. Test B.EQ #0x20 vs B.NE #0x20
        # In ARM64:
        # B.EQ #0x20: 0x54000100 -> b0=00, b1=01, b2=00, b3=54
        # B.NE #0x20: 0x54000101 -> b0=01, b1=01, b2=00, b3=54
        code_beq = bytes.fromhex("00010054")
        code_bne = bytes.fromhex("01010054")

        insn_beq = list(md.disasm(code_beq, 0x1000))[0]
        insn_bne = list(md.disasm(code_bne, 0x1000))[0]

        tokens_beq = mask_arm64_instruction(insn_beq, set())
        tokens_bne = mask_arm64_instruction(insn_bne, set())

        self.assertEqual(tokens_beq, ["00", "??", "??", "54"])
        self.assertEqual(tokens_bne, ["01", "??", "??", "54"])
        # Verify condition code difference is strictly preserved
        self.assertNotEqual(tokens_beq[0], tokens_bne[0])

        # 2. Test ADRP followed by ADD and LDR
        # ADRP X8, #0x1000000
        # ADD X8, X8, #0x710 -> 0x911c4108 (b0=08, b1=41, b2=1c, b3=91)
        # LDR X0, [X8, #0x28] -> 0xf9401500 (b0=00, b1=15, b2=40, b3=f9)
        code_seq = bytes.fromhex("285904d008411c91001540f9")
        insns = list(md.disasm(code_seq, 0x1000))
        adrp_regs = set()

        t_adrp = mask_arm64_instruction(insns[0], adrp_regs)
        self.assertIn("x8", adrp_regs)
        self.assertEqual(t_adrp, ["??", "??", "??", "??"])

        t_add = mask_arm64_instruction(insns[1], adrp_regs)
        # Register byte 0 (08) and opcode byte 3 (91) preserved, immediate bytes 1-2 masked
        self.assertEqual(t_add, ["08", "??", "??", "91"])

        t_ldr = mask_arm64_instruction(insns[2], adrp_regs)
        # Register byte 0 (00) and opcode byte 3 (F9) preserved, immediate offset masked
        self.assertEqual(t_ldr, ["00", "??", "??", "F9"])

    def test_03_find_pointer_chains_3_levels(self):
        """Verify find_pointer_chains discovers 3-level static pointer paths from Module __DATA to dynamic heap address."""
        # Simulated layout:
        # Module: UnityFramework at 0x100000000, __DATA at 0x100005000 - 0x100008000
        # Target address on dynamic heap: 0x160000030
        # Pointer 1 (Static in UnityFramework): at 0x100005108 -> points to 0x140000000 (offset: 0x10 for next level)
        # Pointer 2 (Heap): at 0x140000010 -> points to 0x150000000 (offset: 0xB8 for next level)
        # Pointer 3 (Heap): at 0x1500000B8 -> points to 0x160000000 (offset: 0x30 to target!)

        target_address = "0x160000030"

        mock_modules = [
            {
                "name": "UnityFramework",
                "base_address": 0x100000000,
                "data_segments": [("__DATA", 0x100005000, 0x100008000)],
            }
        ]

        # Construct memory chunks
        chunk_static = bytearray(0x200)
        struct.pack_into("<Q", chunk_static, 0x108, 0x140000000)

        chunk_heap1 = bytearray(0x100)
        struct.pack_into("<Q", chunk_heap1, 0x10, 0x150000000)

        chunk_heap2 = bytearray(0x200)
        struct.pack_into("<Q", chunk_heap2, 0xB8, 0x160000000)

        mock_memory = {
            0x100005000: bytes(chunk_static),
            0x140000000: bytes(chunk_heap1),
            0x150000000: bytes(chunk_heap2),
        }

        res = find_pointer_chains(
            debugger=self.debugger,
            pid=os.getpid(),
            target_address=target_address,
            max_depth=4,
            max_offset=4096,
            mock_modules=mock_modules,
            mock_memory=mock_memory,
        )

        self.assertEqual(res["status"], "success")
        self.assertGreaterEqual(res["total_chains"], 1)

        chain = res["chains_found"][0]
        self.assertEqual(chain["base_module"], "UnityFramework")
        self.assertEqual(chain["base_offset"], "0x00005108")
        self.assertEqual(chain["offsets"], ["0x10", "0xB8", "0x30"])
        self.assertEqual(
            chain["readable_path"],
            "UnityFramework + 0x00005108 -> +0x10 -> +0xB8 -> +0x30",
        )
        print("\n--- Pointer Chain Discovery Success ---")
        print(f"Path: {chain['readable_path']}")

    def test_04_hook_and_log_method_telemetry_structure(self):
        """Verify hook_and_log_method data structure formatting and non-blocking return."""
        mock_invocations = [
            {
                "invocation_id": 1,
                "timestamp_ms": 1730960000000,
                "execution_duration_ms": 0.042,
                "args": {
                    "x0": "0x160000000",
                    "x1": "0x1",
                    "x2": "0x0",
                    "x3": "0x102b50000",
                    "x4": "0x0",
                    "x5": "0x0",
                    "x6": "0x0",
                    "x7": "0x0",
                },
                "float_args": {"d0": 1.0, "d1": 0.0},
                "return_value": "0x1",
                "backtrace": [
                    "0x102b49000 UnityFramework!Update + 0x120",
                    "0x102b50000 UnityFramework!Tick + 0x40",
                ],
            }
        ]

        res = hook_and_log_method(
            debugger=self.debugger,
            pid=os.getpid(),
            function_address="0x102b48000",
            capture_count=1,
            timeout_seconds=2.0,
            mock_invocations=mock_invocations,
        )

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["captured_invocations"], 1)
        self.assertEqual(res["function_address"], "0x102b48000")
        self.assertEqual(res["invocations"][0]["args"]["x0"], "0x160000000")
        self.assertEqual(res["invocations"][0]["return_value"], "0x1")
        self.assertEqual(len(res["invocations"][0]["backtrace"]), 2)


if __name__ == "__main__":
    unittest.main()
