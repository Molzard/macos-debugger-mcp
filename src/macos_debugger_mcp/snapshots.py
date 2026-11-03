"""Heap snapshot capture and differencing engine for memory debugging."""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# In-memory snapshot store: {snapshot_name: {base_address: (size, raw_bytes)}}
_snapshots: Dict[str, Dict[int, bytes]] = {}

DATA_TYPE_FORMATS = {
    "int32": ("<i", 4),
    "uint32": ("<I", 4),
    "int64": ("<q", 8),
    "uint64": ("<Q", 8),
    "float": ("<f", 4),
    "double": ("<d", 8),
}


def take_memory_snapshot(
    debugger: Any,
    pid: int,
    snapshot_name: str,
    max_total_mb: int = 128,
) -> Dict[str, Any]:
    """Capture all writable memory regions (heap rw-) of target process into memory."""
    regions_info = debugger.inspect_memory_regions(pid, limit=1000)
    regions = regions_info.get("regions", [])

    snapshot: Dict[int, bytes] = {}
    total_bytes = 0
    max_bytes = max_total_mb * 1024 * 1024

    for r in regions:
        # Filter for writable regions
        if "w" not in r.get("protection_str", ""):
            continue

        addr_str = r.get("start")
        size = r.get("size", 0)
        if not addr_str or size <= 0:
            continue

        base_addr = int(addr_str, 16) if isinstance(addr_str, str) else addr_str

        # Skip oversized or virtual mapping regions
        chunk_size = min(size, 2 * 1024 * 1024)
        if total_bytes + chunk_size > max_bytes:
            break

        try:
            res = debugger.read_memory(pid, hex(base_addr), chunk_size, fmt="hex")
            hex_data = res.get("hex", "")
            raw = bytes.fromhex(hex_data)
            if raw:
                snapshot[base_addr] = raw
                total_bytes += len(raw)
        except Exception:
            continue

    _snapshots[snapshot_name] = snapshot

    return {
        "status": "success",
        "snapshot_name": snapshot_name,
        "pid": pid,
        "regions_captured": len(snapshot),
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / (1024 * 1024), 2),
    }


def diff_memory_snapshots(
    snapshot_a: str,
    snapshot_b: str,
    data_type: Literal["int32", "uint32", "int64", "uint64", "float", "double"] = "int32",
    filter_type: Literal["changed", "unchanged", "increased", "decreased", "exact_difference"] = "changed",
    diff_value: Optional[float] = None,
    max_results: int = 50,
) -> Dict[str, Any]:
    """Compare two memory snapshots to isolate mutated or stable addresses."""
    if snapshot_a not in _snapshots:
        raise KeyError(f"Snapshot '{snapshot_a}' does not exist in store")
    if snapshot_b not in _snapshots:
        raise KeyError(f"Snapshot '{snapshot_b}' does not exist in store")

    snap_a = _snapshots[snapshot_a]
    snap_b = _snapshots[snapshot_b]

    fmt, item_size = DATA_TYPE_FORMATS.get(data_type, ("<i", 4))
    matches: List[Dict[str, Any]] = []

    # Find common base addresses
    common_bases = set(snap_a.keys()).intersection(snap_b.keys())

    for base in sorted(common_bases):
        buf_a = snap_a[base]
        buf_b = snap_b[base]
        limit = min(len(buf_a), len(buf_b))

        # Align to item_size
        limit -= limit % item_size

        for offset in range(0, limit, item_size):
            val_a = struct.unpack_from(fmt, buf_a, offset)[0]
            val_b = struct.unpack_from(fmt, buf_b, offset)[0]

            matched = False
            delta = val_b - val_a

            if filter_type == "changed":
                matched = (val_a != val_b)
            elif filter_type == "unchanged":
                matched = (val_a == val_b)
            elif filter_type == "increased":
                matched = (val_b > val_a)
            elif filter_type == "decreased":
                matched = (val_b < val_a)
            elif filter_type == "exact_difference":
                if diff_value is not None:
                    matched = abs(delta - diff_value) < 1e-4

            if matched:
                target_addr = base + offset
                matches.append({
                    "address": f"0x{target_addr:x}",
                    "value_before": val_a,
                    "value_after": val_b,
                    "diff": delta,
                })
                if len(matches) >= max_results:
                    break

        if len(matches) >= max_results:
            break

    return {
        "status": "success",
        "snapshot_a": snapshot_a,
        "snapshot_b": snapshot_b,
        "data_type": data_type,
        "filter_type": filter_type,
        "diff_value": diff_value,
        "total_matches": len(matches),
        "matches": matches,
    }
