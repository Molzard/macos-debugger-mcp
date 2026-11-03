"""MCP Server definitions for macos-debugger-mcp."""

from __future__ import annotations

import sys
from typing import Any, Literal, Optional, Union

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore

from .debugger import MacOSDebugger

server = MCPServer("macos-debugger-mcp")
_debugger = MacOSDebugger()


@server.tool()
def list_target_processes(name_filter: Optional[str] = None) -> list[dict[str, Any]]:
    """Enumerate running processes on macOS with PID, name, architecture, and bundle ID.

    Args:
        name_filter: Optional substring to filter process name or path.
    """
    return _debugger.list_processes(name_filter)


@server.tool()
def attach_debugger(pid: int) -> dict[str, Any]:
    """Attach debugger to running process via PID.

    Args:
        pid: Process ID of target application.
    """
    return _debugger.attach(pid)


@server.tool()
def inspect_memory_regions(
    pid: int, limit: int = 100, offset: int = 0
) -> dict[str, Any]:
    """Retrieve virtual memory segments (Mach-O headers, text, data, heap) and page protection flags (r-x, rw-, etc.).

    Args:
        pid: Target process ID.
        limit: Maximum number of memory regions to return (default: 100).
        offset: Number of initial memory regions to skip (pagination).
    """
    return _debugger.inspect_memory_regions(pid, limit=limit, offset=offset)


@server.tool()
def search_memory(
    pid: int,
    value: Union[str, int, float],
    data_type: Literal["int32", "int64", "float", "double", "bytes", "string"] = "int32",
    max_results: int = 50,
    region_type: Literal["all", "writable"] = "writable",
) -> dict[str, Any]:
    """Scan memory regions (e.g. writable heap rw-) of a target process for specific values or byte patterns.

    Args:
        pid: Target process ID.
        value: Value or byte sequence to search for.
        data_type: Data type format ('int32', 'int64', 'float', 'double', 'bytes', 'string').
        max_results: Maximum candidate addresses to return (default: 50).
        region_type: Scope of regions to scan ('writable' or 'all', default: 'writable').
    """
    return _debugger.search_memory(
        pid,
        value,
        data_type=data_type,
        max_results=max_results,
        region_type=region_type,
    )


@server.tool()
def read_memory_bytes(
    pid: int,
    address: str,
    size: int,
    format: Literal["hex", "int32", "int64", "float", "double", "string"] = "hex",
) -> dict[str, Any]:
    """Read structured data from process memory address.

    Args:
        pid: Target process ID.
        address: Memory address in hex (e.g. '0x104b28000').
        size: Number of bytes to read.
        format: Format to unpack data into ('hex', 'int32', 'int64', 'float', 'double', 'string').
    """
    return _debugger.read_memory(pid, address, size, fmt=format)


@server.tool()
def write_memory_bytes(
    pid: int,
    address: str,
    bytes_hex: str,
) -> dict[str, Any]:
    """Modify memory state at a given address (for QA simulation, live patching).

    Args:
        pid: Target process ID.
        address: Memory address in hex (e.g. '0x104b28000').
        bytes_hex: Hex-encoded byte string to write (e.g. '9090' or '48656c6c6f').
    """
    return _debugger.write_memory(pid, address, bytes_hex)


@server.tool()
def set_watchpoint(
    address: str,
    size: Literal[1, 2, 4, 8] = 4,
    watch_type: Literal["read", "write", "read_write"] = "write",
) -> dict[str, Any]:
    """Set an ARM64 hardware watchpoint on an address to observe when value changes or is accessed.

    Args:
        address: Memory address in hex.
        size: Watchpoint size in bytes (1, 2, 4, or 8).
        watch_type: Access condition to trigger watchpoint ('read', 'write', or 'read_write').
    """
    return _debugger.set_watchpoint(address, size, watch_type)


def main() -> None:
    """Run the MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
