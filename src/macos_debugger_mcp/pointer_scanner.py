"""Multi-Level Static Pointer Scanner (find_pointer_chains) for macOS ARM64."""

from __future__ import annotations

import bisect
import ctypes
import os
import struct
from typing import Any, Dict, List, Optional, Set, Tuple

from .utils import parse_address, strip_pac

_libc = ctypes.CDLL(None)
TASK_DYLD_INFO = 17


class task_dyld_info(ctypes.Structure):
    _fields_ = [
        ("all_image_info_addr", ctypes.c_uint64),
        ("all_image_info_size", ctypes.c_uint64),
        ("all_image_info_format", ctypes.c_int32),
    ]


def get_loaded_modules_and_data_segments(
    debugger: Any, pid: int
) -> List[Dict[str, Any]]:
    """Traverse target process dyld image list and extract static __DATA segments."""
    modules: List[Dict[str, Any]] = []

    # 1. Try native Mach TASK_DYLD_INFO
    try:
        task = debugger.get_task_port(pid) if hasattr(debugger, "get_task_port") else 0
        if task:
            info = task_dyld_info()
            count = ctypes.c_uint32(ctypes.sizeof(info) // 4)
            kr = _libc.task_info(task, TASK_DYLD_INFO, ctypes.byref(info), ctypes.byref(count))
            if kr == 0 and info.all_image_info_addr > 0:
                header_raw = debugger.read_raw_bytes(pid, info.all_image_info_addr, 16)
                if len(header_raw) >= 16:
                    _, img_count, img_array = struct.unpack("<IIQ", header_raw)
                    img_array = strip_pac(img_array)

                    for i in range(min(img_count, 500)):
                        info_raw = debugger.read_raw_bytes(pid, img_array + i * 24, 24)
                        if len(info_raw) < 24:
                            break
                        load_addr, path_ptr, _ = struct.unpack("<QQQ", info_raw)
                        load_addr = strip_pac(load_addr)
                        path_ptr = strip_pac(path_ptr)

                        if load_addr == 0 or path_ptr == 0:
                            continue

                        try:
                            path_raw = debugger.read_raw_bytes(pid, path_ptr, 256)
                            null_idx = path_raw.find(b"\x00")
                            if null_idx != -1:
                                path_raw = path_raw[:null_idx]
                            mod_path = path_raw.decode("utf-8", errors="replace")
                            mod_name = os.path.basename(mod_path)
                        except Exception:
                            mod_name = f"Module_0x{load_addr:x}"

                        # Parse Mach-O 64-bit header at load_addr to locate __DATA segments
                        static_ranges = []
                        try:
                            mach_hdr = debugger.read_raw_bytes(pid, load_addr, 32)
                            if len(mach_hdr) >= 32:
                                magic, _, _, _, ncmds, sizeofcmds, _, _ = struct.unpack("<IIIIIIII", mach_hdr)
                                if magic in (0xFEEDFACF, 0xFEEDFACE):
                                    cmd_offset = load_addr + 32
                                    slide = 0
                                    cmd_data = debugger.read_raw_bytes(pid, cmd_offset, min(sizeofcmds, 16384))
                                    c_off = 0
                                    for _ in range(ncmds):
                                        if c_off + 8 > len(cmd_data):
                                            break
                                        cmd, cmdsize = struct.unpack_from("<II", cmd_data, c_off)
                                        if cmd == 0x19:  # LC_SEGMENT_64
                                            seg_hdr = cmd_data[c_off : c_off + 72]
                                            if len(seg_hdr) >= 72:
                                                _, _, segname_raw, vmaddr, vmsize, _, _, _, _, _, _ = struct.unpack(
                                                    "<II16sQQQQiiII", seg_hdr
                                                )
                                                segname = segname_raw.split(b"\x00")[0].decode("latin1", errors="replace")
                                                if segname == "__TEXT":
                                                    slide = load_addr - vmaddr
                                                if "DATA" in segname:
                                                    s_start = vmaddr + slide
                                                    s_end = s_start + vmsize
                                                    static_ranges.append((segname, s_start, s_end))
                                        c_off += cmdsize
                        except Exception:
                            pass

                        if static_ranges:
                            modules.append({
                                "name": mod_name,
                                "base_address": load_addr,
                                "data_segments": static_ranges,
                            })
    except Exception:
        pass

    # 2. Fallback: inspect memory regions for non-heap writable/const regions
    if not modules:
        try:
            regions_info = debugger.inspect_memory_regions(pid, limit=1000)
            for r in regions_info.get("regions", []):
                prot = r.get("protection", "")
                tag = r.get("user_tag", "")
                if "w" in prot or "r" in prot:
                    if "DYLIB" in tag or "STATIC" in tag:
                        start = int(r.get("start_address", "0"), 16)
                        size = r.get("size_bytes", 0)
                        modules.append({
                            "name": tag,
                            "base_address": start,
                            "data_segments": [(tag, start, start + size)],
                        })
        except Exception:
            pass

    return modules


def find_pointer_chains(
    debugger: Any,
    pid: int,
    target_address: str,
    max_depth: int = 4,
    max_offset: int = 4096,
    module_name: Optional[str] = None,
    mock_modules: Optional[List[Dict[str, Any]]] = None,
    mock_memory: Optional[Dict[int, bytes]] = None,
) -> Dict[str, Any]:
    """Find permanent static pointer paths (Module Base + Multi-Level Offsets) pointing to dynamic target_address."""
    target_addr = strip_pac(parse_address(target_address))
    clamped_depth = max(1, min(max_depth, 6))
    clamped_offset = max(0, min(max_offset, 0x10000))

    # 1. Discover modules and static data segments
    modules = mock_modules or get_loaded_modules_and_data_segments(debugger, pid)

    # Filter modules if module_name requested
    if module_name:
        filtered = [m for m in modules if module_name.lower() in m["name"].lower()]
        if filtered:
            modules = filtered

    def is_static_address(addr: int) -> Optional[Tuple[str, int]]:
        for m in modules:
            m_name = m["name"]
            m_base = m["base_address"]
            for _, s_start, s_end in m["data_segments"]:
                if s_start <= addr < s_end:
                    return m_name, addr - m_base
        return None

    # 2. Read pointers across static segments and writable heap regions
    indexed_pointers: List[Tuple[int, int]] = []

    if mock_memory is not None:
        # Mock mode for unit tests
        for base, data in mock_memory.items():
            for off in range(0, len(data) - 7, 8):
                val = strip_pac(struct.unpack_from("<Q", data, off)[0])
                if val > 0x1000:
                    indexed_pointers.append((base + off, val))
    else:
        # Read from target process regions
        regions_info = debugger.inspect_memory_regions(pid, limit=1000)
        PAGE_SIZE = 16384
        CHUNK_SIZE = 1024 * 1024  # 1 MB dynamic chunk
        prealloc_buf = ctypes.create_string_buffer(CHUNK_SIZE)

        for r in regions_info.get("regions", []):
            prot = r.get("protection", "")
            if "w" not in prot and not any(
                s_start <= int(r.get("start_address", "0"), 16) < s_end
                for m in modules
                for _, s_start, s_end in m["data_segments"]
            ):
                continue

            start_addr = int(r.get("start_address", "0"), 16)
            size = r.get("size_bytes", 0)
            end_addr = start_addr + size

            curr = start_addr
            while curr < end_addr:
                to_read = min(CHUNK_SIZE, end_addr - curr)
                if to_read > PAGE_SIZE:
                    to_read = (to_read // PAGE_SIZE) * PAGE_SIZE

                try:
                    if hasattr(debugger, "read_raw_bytes"):
                        chunk = debugger.read_raw_bytes(pid, curr, to_read, preallocated_buf=prealloc_buf)
                    else:
                        res = debugger.read_memory(pid, hex(curr), to_read, fmt="hex")
                        hex_data = res.get("hex", "")
                        chunk = bytes.fromhex(hex_data) if hex_data else b""
                except Exception:
                    curr += to_read
                    continue

                for off in range(0, len(chunk) - 7, 8):
                    val = strip_pac(struct.unpack_from("<Q", chunk, off)[0])
                    if val > 0x1000:
                        indexed_pointers.append((curr + off, val))

                curr += to_read

    # 3. Sort pointers by value for logarithmic bisect lookup
    indexed_pointers.sort(key=lambda x: x[1])
    val_keys = [p[1] for p in indexed_pointers]

    # 4. BFS Backward Pointer Traversal
    # State tuple: (current_target_address, list_of_hex_offsets, depth, visited_set)
    queue: List[Tuple[int, List[str], int, Set[int]]] = [(target_addr, [], 0, {target_addr})]
    chains_found: List[Dict[str, Any]] = []
    seen_chain_signatures: Set[str] = set()

    while queue and len(chains_found) < 20:
        curr_target, current_offsets, depth, visited = queue.pop(0)
        if depth >= clamped_depth:
            continue

        low_val = max(0, curr_target - clamped_offset)
        high_val = curr_target

        idx_start = bisect.bisect_left(val_keys, low_val)
        idx_end = bisect.bisect_right(val_keys, high_val)

        for i in range(idx_start, idx_end):
            p_addr, p_val = indexed_pointers[i]
            if p_addr in visited:
                continue

            offset = curr_target - p_val
            hex_offset = f"0x{offset:X}"
            new_offsets = [hex_offset] + current_offsets

            static_hit = is_static_address(p_addr)
            if static_hit is not None:
                mod_name, base_offset = static_hit
                readable_offsets = " -> ".join(f"+{o}" for o in new_offsets)
                chain_sig = f"{mod_name}:0x{base_offset:x}:{readable_offsets}"

                if chain_sig not in seen_chain_signatures:
                    seen_chain_signatures.add(chain_sig)
                    chains_found.append({
                        "base_module": mod_name,
                        "base_offset": f"0x{base_offset:08x}",
                        "offsets": new_offsets,
                        "readable_path": f"{mod_name} + 0x{base_offset:08x} -> {readable_offsets}",
                    })
                    if len(chains_found) >= 20:
                        break
            else:
                if depth + 1 < clamped_depth and len(chains_found) < 20:
                    new_visited = set(visited)
                    new_visited.add(p_addr)
                    queue.append((p_addr, new_offsets, depth + 1, new_visited))

    return {
        "status": "success",
        "target_address": f"0x{target_addr:x}",
        "max_depth": clamped_depth,
        "max_offset": clamped_offset,
        "total_chains": len(chains_found),
        "chains_found": chains_found,
    }
