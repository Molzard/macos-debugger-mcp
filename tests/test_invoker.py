"""Unit tests for in-process method invocation tool (invoke_native_method)."""

from __future__ import annotations

import ctypes
import os
import sys
import unittest

# Ensure package is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from macos_debugger_mcp.debugger import MacOSDebugger
from macos_debugger_mcp.invoker import invoke_native_method


class TestInvoker(unittest.TestCase):
    def setUp(self):
        self.debugger = MacOSDebugger()
        self.pid = os.getpid()
        self.libc = ctypes.CDLL(None)

    def test_invoke_getpid_local(self):
        """Test invoking getpid() on current process and verifying return value."""
        getpid_ptr = ctypes.cast(self.libc.getpid, ctypes.c_void_p).value
        assert getpid_ptr is not None
        getpid_addr = hex(getpid_ptr)
        resp = invoke_native_method(
            self.debugger,
            self.pid,
            getpid_addr,
            args=[],
            return_type="int32",
            timeout=5.0,
        )

        self.assertEqual(resp["status"], "success")
        self.assertEqual(resp["pid"], self.pid)
        self.assertEqual(resp["engine"], "frida")
        self.assertEqual(resp["return_type"], "int32")
        self.assertEqual(resp["result"], self.pid)
        print(f"[PASS] getpid invocation: expected={self.pid}, got={resp['result']}")

    def test_invoke_puts_with_string_arg(self):
        """Test invoking puts(char*) with string argument."""
        puts_ptr = ctypes.cast(self.libc.puts, ctypes.c_void_p).value
        assert puts_ptr is not None
        puts_addr = hex(puts_ptr)
        test_msg = "MCP Invoker: Native String Parameter Verification"
        resp = invoke_native_method(
            self.debugger,
            self.pid,
            puts_addr,
            args=[{"type": "string", "value": test_msg}],
            return_type="int32",
            timeout=5.0,
        )

        self.assertEqual(resp["status"], "success")
        self.assertGreaterEqual(resp["result"], 0)
        print(f"[PASS] puts invocation returned status code: {resp['result']}")

    def test_non_executable_address_guard(self):
        """Test safety guard rejecting addresses outside executable code regions."""
        # 0x1000 is unmapped / non-executable
        bad_addr = "0x1000"
        with self.assertRaises(PermissionError):
            invoke_native_method(
                self.debugger,
                self.pid,
                bad_addr,
                args=[],
                return_type="void",
                timeout=2.0,
            )
        print("[PASS] Safety guard correctly blocked invalid/non-executable address.")


if __name__ == "__main__":
    unittest.main()
