"""ARM64 NEON & GCD Native Acceleration Bridge for macOS Debugger MCP."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from typing import Any, List, Optional, Tuple, Union

_accel_lib: Optional[ctypes.CDLL] = None
_load_attempted: bool = False

# Mapping comparison types to C constants
COMP_TYPES = {
    "changed": 0,
    "unchanged": 1,
    "increased": 2,
    "decreased": 3,
    "exact_difference": 4,
}

# Mapping data types to C constants
DATA_TYPES = {
    "int32": 0,
    "uint32": 1,
    "int64": 2,
    "uint64": 3,
    "float": 4,
    "double": 5,
}


def load_accel_core() -> Optional[ctypes.CDLL]:
    """Transparently compile (if needed) and load libmcp_accel.dylib for ARM64 NEON/GCD.

    Returns the CDLL instance or None if unavailable/unsupported.
    """
    global _accel_lib, _load_attempted
    if _load_attempted:
        return _accel_lib

    _load_attempted = True

    # Only supported on ARM64 macOS (Apple Silicon)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    native_dir = os.path.join(os.path.dirname(__file__), "native")
    dylib_path = os.path.join(native_dir, "libmcp_accel.dylib")
    src_c_path = os.path.join(native_dir, "accel_core.c")

    # If dylib is missing, auto-compile once via clang
    if not os.path.isfile(dylib_path) and os.path.isfile(src_c_path):
        try:
            cmd = [
                "clang",
                "-O3",
                "-arch",
                "arm64",
                "-shared",
                "-fPIC",
                src_c_path,
                "-o",
                dylib_path,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if res.returncode != 0:
                return None
        except Exception:
            return None

    if not os.path.isfile(dylib_path):
        return None

    try:
        lib = ctypes.CDLL(dylib_path)

        # 1. aob_scan_neon
        lib.aob_scan_neon.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        lib.aob_scan_neon.restype = ctypes.c_int64

        # 2. aob_scan_all_neon
        lib.aob_scan_all_neon.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        lib.aob_scan_all_neon.restype = ctypes.c_size_t

        # 3. diff_snapshots_neon
        lib.diff_snapshots_neon.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        lib.diff_snapshots_neon.restype = ctypes.c_size_t

        # 4. diff_snapshots_gcd
        lib.diff_snapshots_gcd.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_size_t,
        ]
        lib.diff_snapshots_gcd.restype = ctypes.c_size_t

        _accel_lib = lib
        return _accel_lib
    except Exception:
        return None


def _get_buffer_pointer(buf: Any) -> Tuple[ctypes.c_void_p, int]:
    """Extract raw memory address and length from bytes, bytearray, memoryview, or int pointer."""
    if isinstance(buf, bytes):
        return ctypes.cast(ctypes.c_char_p(buf), ctypes.c_void_p), len(buf)
    elif isinstance(buf, bytearray):
        buf_c = (ctypes.c_char * len(buf)).from_buffer(buf)
        return ctypes.cast(ctypes.addressof(buf_c), ctypes.c_void_p), len(buf)
    elif isinstance(buf, memoryview):
        buf_c = (ctypes.c_char * len(buf)).from_buffer(buf)
        return ctypes.cast(ctypes.addressof(buf_c), ctypes.c_void_p), len(buf)
    elif isinstance(buf, int):
        return ctypes.c_void_p(buf), 0
    else:
        # Fallback to string buffer or cast
        c_ptr = ctypes.cast(buf, ctypes.c_void_p)
        return c_ptr, getattr(buf, "size", 0)


def aob_scan_neon(
    data: Union[bytes, bytearray, memoryview, int],
    data_len: int,
    pattern: bytes,
    mask: bytes,
) -> int:
    """Find first matching offset of wildcard pattern via NEON. Returns -1 if not found."""
    lib = load_accel_core()
    pat_len = len(pattern)
    if not lib or data_len < pat_len:
        # Pure Python fallback
        if isinstance(data, int):
            return -1
        b_data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        for i in range(data_len - pat_len + 1):
            matched = True
            for p in range(pat_len):
                if (b_data[i + p] & mask[p]) != pattern[p]:
                    matched = False
                    break
            if matched:
                return i
        return -1

    ptr, _ = _get_buffer_pointer(data)
    idx = lib.aob_scan_neon(ptr, data_len, pattern, mask, pat_len)
    return int(idx)


def aob_scan_all_neon(
    data: Union[bytes, bytearray, memoryview, int],
    data_len: int,
    pattern: bytes,
    mask: bytes,
    max_results: int = 50,
) -> List[int]:
    """Find all matching offsets of wildcard pattern via NEON up to max_results."""
    lib = load_accel_core()
    pat_len = len(pattern)
    if not lib or data_len < pat_len:
        # Pure Python fallback
        if isinstance(data, int):
            return []
        b_data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        matches = []
        for i in range(data_len - pat_len + 1):
            matched = True
            for p in range(pat_len):
                if (b_data[i + p] & mask[p]) != pattern[p]:
                    matched = False
                    break
            if matched:
                matches.append(i)
                if len(matches) >= max_results:
                    break
        return matches

    ptr, _ = _get_buffer_pointer(data)
    out_arr = (ctypes.c_uint64 * max_results)()
    count = lib.aob_scan_all_neon(ptr, data_len, pattern, mask, pat_len, out_arr, max_results)
    return [int(out_arr[i]) for i in range(count)]


def diff_snapshots_neon(
    old_buf: Any,
    new_buf: Any,
    size: int,
    comp_type: str = "changed",
    data_type: str = "int32",
    target_val: float = 0.0,
    max_results: int = 50,
    use_gcd: bool = True,
) -> List[int]:
    """Compare two snapshot buffers via ARM64 NEON & GCD fast page-skip.

    Returns list of relative byte offsets that matched the filter criteria.
    """
    lib = load_accel_core()
    c_comp = COMP_TYPES.get(comp_type.lower(), 0)
    c_type = DATA_TYPES.get(data_type.lower(), 0)

    if not lib:
        # Python fallback if native dylib is unavailable
        import struct
        fmt_map = {
            "int32": ("<i", 4),
            "uint32": ("<I", 4),
            "int64": ("<q", 8),
            "uint64": ("<Q", 8),
            "float": ("<f", 4),
            "double": ("<d", 8),
        }
        fmt, item_size = fmt_map.get(data_type.lower(), ("<i", 4))
        limit = (size // item_size) * item_size
        matches = []

        # Read into memoryview or bytes
        mv_a = old_buf if isinstance(old_buf, memoryview) else memoryview(old_buf)
        mv_b = new_buf if isinstance(new_buf, memoryview) else memoryview(new_buf)

        for off in range(0, limit, item_size):
            val_a = struct.unpack_from(fmt, mv_a, off)[0]
            val_b = struct.unpack_from(fmt, mv_b, off)[0]
            matched = False
            delta = val_b - val_a

            if comp_type == "changed":
                matched = (val_a != val_b)
            elif comp_type == "unchanged":
                matched = (val_a == val_b)
            elif comp_type == "increased":
                matched = (val_b > val_a)
            elif comp_type == "decreased":
                matched = (val_b < val_a)
            elif comp_type == "exact_difference":
                matched = abs(delta - target_val) < 1e-4

            if matched:
                matches.append(off)
                if len(matches) >= max_results:
                    break
        return matches

    ptr_old, _ = _get_buffer_pointer(old_buf)
    ptr_new, _ = _get_buffer_pointer(new_buf)
    out_arr = (ctypes.c_uint64 * max_results)()

    diff_func = lib.diff_snapshots_gcd if (use_gcd and size >= 2 * 1024 * 1024) else lib.diff_snapshots_neon
    count = diff_func(
        ptr_old,
        ptr_new,
        size,
        c_comp,
        c_type,
        ctypes.c_double(target_val),
        out_arr,
        max_results,
    )

    return [int(out_arr[i]) for i in range(count)]
