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


@server.tool()
def resolve_object_at_address(
    pid: int,
    address: str,
    metadata_path: Optional[str] = None,
) -> dict[str, Any]:
    """Reconstruct semantic IL2CPP object graph (class, fields, offsets, values) at memory address.

    Args:
        pid: Process ID of running Unity application.
        address: Target heap memory address in hex (e.g. '0x104b28190').
        metadata_path: Optional path to Unity global-metadata.dat file.
    """
    from .rtti_resolver import MachVMAccess, RTTIResolver
    from .il2cpp_metadata import Il2CppMetadataParser

    default_meta = os.path.expanduser(
        "~/Library/Containers/io.playcover.PlayCover/Applications/com.rovio.baba.app/Data/Managed/Metadata/global-metadata.dat"
    )
    meta_file = metadata_path or default_meta
    parser = Il2CppMetadataParser(meta_file)
    vm = MachVMAccess(pid)
    resolver = RTTIResolver(parser, vm)
    result = resolver.resolve_object_at_address(address)
    vm.close()
    return result


@server.tool()
def find_active_instances_of_class(
    pid: int,
    class_name: str,
    max_results: int = 50,
    metadata_path: Optional[str] = None,
) -> dict[str, Any]:
    """Scan process writable memory (heap) for live instances of a given IL2CPP class.

    Args:
        pid: Process ID of running Unity application.
        class_name: Target class name or full namespace (e.g. 'CardButton').
        max_results: Maximum matching object instances to return (default: 50).
        metadata_path: Optional path to Unity global-metadata.dat file.
    """
    from .rtti_resolver import MachVMAccess, RTTIResolver
    from .il2cpp_metadata import Il2CppMetadataParser

    default_meta = os.path.expanduser(
        "~/Library/Containers/io.playcover.PlayCover/Applications/com.rovio.baba.app/Data/Managed/Metadata/global-metadata.dat"
    )
    meta_file = metadata_path or default_meta
    parser = Il2CppMetadataParser(meta_file)
    vm = MachVMAccess(pid)
    resolver = RTTIResolver(parser, vm)
    instances = resolver.find_active_instances(class_name, max_results=max_results)
    vm.close()
    return {
        "status": "success",
        "pid": pid,
        "class_name": class_name,
        "count": len(instances),
        "instances": instances,
    }


@server.tool()
def take_memory_snapshot(
    pid: int,
    snapshot_name: str,
) -> dict[str, Any]:
    """Capture a point-in-time snapshot of all writable memory regions (heap) for diffing.

    Args:
        pid: Process ID of running application.
        snapshot_name: Identifier name for the snapshot (e.g. 'before_shot', 'stage_start').
    """
    from .snapshots import take_memory_snapshot as _snap
    return _snap(_debugger, pid, snapshot_name)


@server.tool()
def diff_memory_snapshots(
    snapshot_a: str,
    snapshot_b: str,
    data_type: Literal["int32", "uint32", "int64", "uint64", "float", "double"] = "int32",
    filter_type: Literal["changed", "unchanged", "increased", "decreased", "exact_difference"] = "changed",
    diff_value: Optional[float] = None,
    max_results: int = 50,
) -> dict[str, Any]:
    """Compare two memory snapshots to isolate addresses whose values mutated or remained static.

    Args:
        snapshot_a: Name of initial snapshot (e.g. 'before_shot').
        snapshot_b: Name of subsequent snapshot (e.g. 'after_shot').
        data_type: Unpack type ('int32', 'uint32', 'int64', 'uint64', 'float', 'double').
        filter_type: Mutation rule ('changed', 'unchanged', 'increased', 'decreased', 'exact_difference').
        diff_value: Optional target difference (e.g. -1 to find values decreased by 1).
        max_results: Maximum matching addresses to return (default: 50).
    """
    from .snapshots import diff_memory_snapshots as _diff
    return _diff(
        snapshot_a,
        snapshot_b,
        data_type=data_type,
        filter_type=filter_type,
        diff_value=diff_value,
        max_results=max_results,
    )


@server.tool()
def freeze_address_value(
    pid: int,
    address: str,
    bytes_hex: str,
    interval_ms: int = 100,
) -> dict[str, Any]:
    """Lock a memory address to a fixed value using a high-frequency background daemon thread.

    Args:
        pid: Target process ID.
        address: Memory address in hex (e.g. '0x104b28190').
        bytes_hex: Hex payload to assert repeatedly (e.g. '05000000' or '9090').
        interval_ms: Assertion period in milliseconds (default: 100ms).
    """
    from .freeze_daemon import freeze_address_value as _freeze
    return _freeze(_debugger, pid, address, bytes_hex, interval_ms)


@server.tool()
def unfreeze_address_value(
    freeze_id: str,
) -> dict[str, Any]:
    """Release and terminate active memory freeze assertion worker(s).

    Args:
        freeze_id: Identifier returned by freeze_address_value (or 'all' to stop all workers).
    """
    from .freeze_daemon import unfreeze_address_value as _unfreeze
    return _unfreeze(freeze_id)


@server.tool()
def export_dylib_patch(
    patch_name: str,
    target_framework: str,
    patches: list[dict[str, Any]],
    output_dir: str = "/Users/alfaroo/Sintax/MCP/macos-debugger-mcp/dist_patches",
) -> dict[str, Any]:
    """Generate and compile a standalone ARM64 C/Objective-C dylib patch for PlayCover/macOS.

    Args:
        patch_name: Name of generated patch (e.g. 'InfiniteBirdsPatch').
        target_framework: Name of target framework or binary (e.g. 'UnityFramework').
        patches: Array of patch definitions (offset, patch_bytes, description).
        output_dir: Directory where patch source, Makefile, and compiled .dylib are placed.
    """
    from .dylib_exporter import export_dylib_patch as _export
    return _export(patch_name, target_framework, patches, output_dir)


def main() -> None:
    """Run the MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
