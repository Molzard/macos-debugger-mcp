"""Type conversions, memory unpacking, and Mach error formatting."""

from __future__ import annotations

import struct
from typing import Any

# Mach VM Protection Constants
VM_PROT_NONE = 0x00
VM_PROT_READ = 0x01
VM_PROT_WRITE = 0x02
VM_PROT_EXECUTE = 0x04

# Common Mach Kernel Return Codes
MACH_ERRORS: dict[int, str] = {
    0: "KERN_SUCCESS: Operation succeeded",
    1: "KERN_INVALID_ADDRESS: Specified address does not exist or is unmapped",
    2: "KERN_PROTECTION_FAILURE: Protection violation (memory page not accessible)",
    3: "KERN_NO_SPACE: Address range exceeds process address space",
    4: "KERN_INVALID_ARGUMENT: Invalid argument passed to Mach API",
    5: "KERN_FAILURE: Task port access denied (requires root/sudo, DevToolsSecurity enabled, or target get-task-allow entitlement)",
    8: "KERN_RESOURCE_SHORTAGE: Insufficient Mach resources",
    9: "KERN_NOT_RECEIVER: Port operation denied",
    15: "KERN_INVALID_NAME: Invalid or dead Mach port name",
    46: "KERN_MEMORY_ERROR: Physical memory error",
}

# Common macOS User Tags for VM regions
VM_USER_TAGS: dict[int, str] = {
    1: "VM_MEMORY_MALLOC",
    2: "VM_MEMORY_MALLOC_SMALL",
    3: "VM_MEMORY_MALLOC_LARGE",
    4: "VM_MEMORY_MALLOC_HUGE",
    5: "VM_MEMORY_SBRK",
    6: "VM_MEMORY_REALLOC",
    7: "VM_MEMORY_MALLOC_TINY",
    8: "VM_MEMORY_MALLOC_LARGE_REUSABLE",
    9: "VM_MEMORY_MALLOC_LARGE_REUSED",
    10: "VM_MEMORY_ANALYSIS_TOOL",
    11: "VM_MEMORY_MALLOC_NANO",
    20: "VM_MEMORY_MACH_MSG",
    21: "VM_MEMORY_IOKIT",
    30: "VM_MEMORY_STACK",
    31: "VM_MEMORY_STACK_EMPTY",
    32: "VM_MEMORY_SHARED_PMAP",
    33: "VM_MEMORY_DYLIB",
    34: "VM_MEMORY_OBJC_DISPATCHERS",
    40: "VM_MEMORY_APPKIT",
    54: "VM_MEMORY_FOUNDATION",
    60: "VM_MEMORY_COREGRAPHICS",
    70: "VM_MEMORY_CARBON",
    80: "VM_MEMORY_JAVA",
    90: "VM_MEMORY_SECURITY",
}


def strip_pac(ptr: Any) -> int:
    """Strip Pointer Authentication Codes (PAC) / Top-Byte Ignore (TBI) on ARM64/ARM64e.
    User-space virtual memory addresses on macOS Apple Silicon are capped at 48-bit.
    """
    if ptr is None or not isinstance(ptr, int):
        return 0
    return ptr & 0x00007FFFFFFFFFFF


def parse_address(addr: str | int) -> int:
    """Parse string hex address (e.g. '0x104b28000') or int into integer address."""
    if isinstance(addr, int):
        return addr
    s = addr.strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    try:
        return int(s, 16)
    except ValueError:
        return int(s, 10)


def format_protection(prot: int) -> str:
    """Format VM protection bitmask into Unix-style permissions string ('r-x', 'rw-', etc.)."""
    r = "r" if (prot & VM_PROT_READ) else "-"
    w = "w" if (prot & VM_PROT_WRITE) else "-"
    x = "x" if (prot & VM_PROT_EXECUTE) else "-"
    return f"{r}{w}{x}"


def format_bytes(raw_bytes: bytes, fmt: str) -> Any:
    """Convert raw bytes into structured target format."""
    fmt = fmt.lower()
    if fmt == "hex":
        return raw_bytes.hex()

    if fmt == "int32":
        if len(raw_bytes) < 4:
            raise ValueError(f"int32 requires at least 4 bytes, received {len(raw_bytes)}")
        val = struct.unpack("<i", raw_bytes[:4])[0]
        return val

    if fmt == "int64":
        if len(raw_bytes) < 8:
            raise ValueError(f"int64 requires at least 8 bytes, received {len(raw_bytes)}")
        val = struct.unpack("<q", raw_bytes[:8])[0]
        return val

    if fmt == "float":
        if len(raw_bytes) < 4:
            raise ValueError(f"float requires at least 4 bytes, received {len(raw_bytes)}")
        val = struct.unpack("<f", raw_bytes[:4])[0]
        return val

    if fmt == "double":
        if len(raw_bytes) < 8:
            raise ValueError(f"double requires at least 8 bytes, received {len(raw_bytes)}")
        val = struct.unpack("<d", raw_bytes[:8])[0]
        return val

    if fmt == "string":
        # Decode UTF-8 or fall back to latin-1, terminate at null byte
        null_idx = raw_bytes.find(b"\x00")
        sub = raw_bytes[:null_idx] if null_idx != -1 else raw_bytes
        try:
            return sub.decode("utf-8")
        except UnicodeDecodeError:
            return sub.decode("latin-1", errors="replace")

    raise ValueError(f"Unsupported format: {fmt}. Expected one of ['hex', 'int32', 'int64', 'float', 'double', 'string']")


def format_size(size_bytes: int) -> str:
    """Return human-readable byte size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def mach_error_string(kr: int) -> str:
    """Convert Mach return code to descriptive string."""
    return MACH_ERRORS.get(kr, f"KERN_CODE_{kr}: Unknown Mach error")
