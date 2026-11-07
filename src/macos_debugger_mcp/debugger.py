"""LLDB and native Mach API bridge for macOS ARM64 process debugging."""

from __future__ import annotations

import ctypes
import json
import os
import plistlib
import struct
import subprocess
import sys
from typing import Any, Optional

from .utils import (
    VM_USER_TAGS,
    format_bytes,
    format_protection,
    format_size,
    mach_error_string,
    parse_address,
    strip_pac,
)

# Load native libraries
_libc = ctypes.CDLL(None)
try:
    _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
except OSError:
    _libproc = None

PAGE_SIZE = 16384  # 16 KB page size on macOS Apple Silicon (ARM64)


# Mach structures and prototypes
class vm_region_submap_info_64(ctypes.Structure):
    _fields_ = [
        ("protection", ctypes.c_int32),
        ("max_protection", ctypes.c_int32),
        ("inheritance", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
        ("user_tag", ctypes.c_uint32),
        ("pages_resident", ctypes.c_uint32),
        ("pages_shared_now_private", ctypes.c_uint32),
        ("pages_swapped_out", ctypes.c_uint32),
        ("pages_dirtied", ctypes.c_uint32),
        ("ref_count", ctypes.c_uint32),
        ("shadow_depth", ctypes.c_uint16),
        ("external_pager", ctypes.c_uint8),
        ("share_mode", ctypes.c_uint8),
        ("is_submap", ctypes.c_uint32),
        ("behavior", ctypes.c_int32),
        ("object_id", ctypes.c_uint32),
        ("user_wired_count", ctypes.c_uint16),
        ("pages_reusable", ctypes.c_uint32),
        ("object_id_full", ctypes.c_uint64),
    ]

_mach_task_self = _libc.mach_task_self
_mach_task_self.restype = ctypes.c_uint32

_task_for_pid = _libc.task_for_pid
_task_for_pid.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
_task_for_pid.restype = ctypes.c_int

_mach_port_deallocate = _libc.mach_port_deallocate
_mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_mach_port_deallocate.restype = ctypes.c_int

_mach_vm_region_recurse = getattr(_libc, "mach_vm_region_recurse", None)
if _mach_vm_region_recurse:
    _mach_vm_region_recurse.argtypes = [
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    _mach_vm_region_recurse.restype = ctypes.c_int

_mach_vm_read_overwrite = getattr(_libc, "mach_vm_read_overwrite", None)
if _mach_vm_read_overwrite:
    _mach_vm_read_overwrite.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _mach_vm_read_overwrite.restype = ctypes.c_int

_mach_vm_write = getattr(_libc, "mach_vm_write", None)
if _mach_vm_write:
    _mach_vm_write.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_uint32,
    ]
    _mach_vm_write.restype = ctypes.c_int

_mach_vm_protect = getattr(_libc, "mach_vm_protect", None)
if _mach_vm_protect:
    _mach_vm_protect.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _mach_vm_protect.restype = ctypes.c_int

# Attempt importing LLDB module if available in Python path or standard Xcode paths
lldb = None
_xcode_lldb_paths = [
    "/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Resources/Python",
    "/Library/Developer/CommandLineTools/Library/PrivateFrameworks/LLDB.framework/Resources/Python",
]
for p in _xcode_lldb_paths:
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)

try:
    import lldb  # type: ignore
except Exception:
    lldb = None


class MacOSDebugger:
    """Core macOS debugger bridging LLDB Scripting API and Mach VM kernel calls."""

    def __init__(self) -> None:
        self.active_pid: Optional[int] = None
        self.task_ports: dict[int, int] = {}
        self.lldb_debugger: Any = None
        self.lldb_target: Any = None
        self.lldb_process: Any = None

        if lldb is not None:
            try:
                self.lldb_debugger = lldb.SBDebugger.Create()
                if self.lldb_debugger and self.lldb_debugger.IsValid():
                    self.lldb_debugger.SetAsync(False)
            except Exception:
                self.lldb_debugger = None

    def get_task_port(self, pid: int) -> int:
        """Acquire Mach task port for PID with detailed diagnostic error handling."""
        if pid in self.task_ports:
            return self.task_ports[pid]

        if pid == os.getpid():
            port = _mach_task_self()
            self.task_ports[pid] = port
            return port

        port_out = ctypes.c_uint32(0)
        self_port = _mach_task_self()
        kr = _task_for_pid(self_port, pid, ctypes.byref(port_out))

        if kr != 0:
            err_msg = mach_error_string(kr)
            raise PermissionError(
                f"Failed to acquire task port for PID {pid} (kr={kr}): {err_msg}.\n"
                "macOS Security Diagnostic:\n"
                "1. If debugging external apps, run with root/sudo or ensure Developer Mode is enabled:\n"
                "   sudo DevToolsSecurity -enable\n"
                "2. Non-root users can only debug processes signed with the 'get-task-allow' entitlement\n"
                "   or running with the same UID without Hardened Runtime restrictions."
            )

        self.task_ports[pid] = port_out.value
        return port_out.value

    def cleanup_task_port(self, pid: int) -> bool:
        """Deallocate Mach task port for PID if held."""
        if pid in self.task_ports:
            port = self.task_ports.pop(pid)
            if port != _mach_task_self():
                try:
                    _mach_port_deallocate(_mach_task_self(), port)
                except Exception:
                    pass
            return True
        return False

    def read_raw_bytes(
        self, pid: int, address: int, size: int, preallocated_buf: Optional[Any] = None
    ) -> bytes:
        """Safely read raw memory bytes with preallocated buffer support and 16KB guard-page fallback."""
        if size <= 0:
            return b""
        address = strip_pac(address)
        task = self.get_task_port(pid)
        if not _mach_vm_read_overwrite:
            raise RuntimeError("mach_vm_read_overwrite API not available")

        buf = preallocated_buf if (preallocated_buf is not None and len(preallocated_buf) >= size) else ctypes.create_string_buffer(size)
        buf_addr = ctypes.addressof(buf)
        out_size = ctypes.c_size_t(0)

        kr = _mach_vm_read_overwrite(
            task,
            ctypes.c_uint64(address),
            ctypes.c_uint64(size),
            ctypes.c_uint64(buf_addr),
            ctypes.byref(out_size),
        )
        if kr == 0:
            return bytes(buf.raw[: out_size.value])

        if size > PAGE_SIZE:
            result = bytearray()
            curr = address
            end = address + size
            page_buf = ctypes.create_string_buffer(PAGE_SIZE)
            page_buf_addr = ctypes.addressof(page_buf)
            while curr < end:
                p_read = min(PAGE_SIZE, end - curr)
                p_kr = _mach_vm_read_overwrite(
                    task,
                    ctypes.c_uint64(curr),
                    ctypes.c_uint64(p_read),
                    ctypes.c_uint64(page_buf_addr),
                    ctypes.byref(out_size),
                )
                if p_kr == 0:
                    result.extend(page_buf.raw[: out_size.value])
                else:
                    result.extend(b"\x00" * p_read)
                curr += p_read
            return bytes(result)

        err_desc = mach_error_string(kr)
        raise ValueError(f"Read failed at 0x{address:x} (size {size}): {err_desc}")

    def list_processes(self, name_filter: Optional[str] = None) -> list[dict[str, Any]]:
        """List running macOS processes with PID, name, arch, and bundle ID."""
        if not _libproc:
            raise RuntimeError("libproc.dylib not available on this system")

        PROC_ALL_PIDS = 1
        needed = _libproc.proc_listpids(PROC_ALL_PIDS, 0, None, 0)
        if needed <= 0:
            return []

        count = needed // 4
        pids_arr = (ctypes.c_int * count)()
        ret = _libproc.proc_listpids(PROC_ALL_PIDS, 0, pids_arr, needed)
        if ret <= 0:
            return []

        num_pids = ret // 4
        proc_path_buf = ctypes.create_string_buffer(4096)
        results: list[dict[str, Any]] = []

        is_apple_silicon = os.uname().machine == "arm64"
        filter_lower = name_filter.strip().lower() if name_filter else None

        for i in range(num_pids):
            pid = pids_arr[i]
            if pid <= 0:
                continue

            path_len = _libproc.proc_pidpath(pid, proc_path_buf, 4096)
            if path_len <= 0:
                continue

            try:
                exe_path = proc_path_buf.value.decode("utf-8")
            except UnicodeDecodeError:
                exe_path = proc_path_buf.value.decode("latin-1", errors="replace")

            name = os.path.basename(exe_path)
            if filter_lower and filter_lower not in name.lower() and filter_lower not in exe_path.lower():
                continue

            # Resolve bundle ID if app bundle
            bundle_id: Optional[str] = None
            if ".app" in exe_path:
                idx = exe_path.find(".app")
                app_bundle = exe_path[: idx + 4]
                plist_path = os.path.join(app_bundle, "Contents", "Info.plist")
                if os.path.isfile(plist_path):
                    try:
                        with open(plist_path, "rb") as f:
                            plist_data = plistlib.load(f)
                            bundle_id = plist_data.get("CFBundleIdentifier")
                    except Exception:
                        pass

            # Detect architecture: default to arm64 on Apple Silicon unless translated
            arch = "arm64" if is_apple_silicon else "x86_64"

            results.append({
                "pid": pid,
                "name": name,
                "path": exe_path,
                "architecture": arch,
                "bundle_id": bundle_id,
            })

        results.sort(key=lambda x: x["pid"])
        return results

    def attach(self, pid: int) -> dict[str, Any]:
        """Attach LLDB debugger or obtain native Mach task port for target process."""
        self.active_pid = pid
        backend_used = []
        lldb_attached = False
        mach_attached = False

        # 1. Try native Mach task port
        try:
            self.get_task_port(pid)
            mach_attached = True
            backend_used.append("Mach Kernel VM")
        except Exception as e:
            mach_err = str(e)

        # 2. Try LLDB SBDebugger attach if available
        if self.lldb_debugger and self.lldb_debugger.IsValid():
            try:
                target = self.lldb_debugger.CreateTarget("")
                if target and target.IsValid():
                    listener = lldb.SBListener("mcp.debugger.listener")
                    error = lldb.SBError()
                    process = target.AttachToProcessWithID(listener, pid, error)
                    if error.Success() and process.IsValid():
                        self.lldb_target = target
                        self.lldb_process = process
                        lldb_attached = True
                        backend_used.append("Apple LLDB")
            except Exception:
                pass

        if not mach_attached and not lldb_attached:
            raise PermissionError(
                f"Failed to attach debugger to PID {pid}.\n"
                f"{mach_err if 'mach_err' in locals() else 'Access denied.'}"
            )

        return {
            "status": "attached",
            "pid": pid,
            "backends": backend_used,
            "lldb_available": lldb_attached,
            "mach_port_available": mach_attached,
        }

    def inspect_memory_regions(
        self, pid: int, limit: int = 100, offset: int = 0
    ) -> dict[str, Any]:
        """Enumerate virtual memory map and protection flags for process."""
        task = self.get_task_port(pid)
        if not _mach_vm_region_recurse:
            raise RuntimeError("mach_vm_region_recurse API is unavailable")

        address = ctypes.c_uint64(0)
        size = ctypes.c_uint64(0)
        nesting_depth = ctypes.c_uint32(0)
        info = vm_region_submap_info_64()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)

        regions: list[dict[str, Any]] = []
        total_found = 0
        skipped = 0

        while True:
            kr = _mach_vm_region_recurse(
                task,
                ctypes.byref(address),
                ctypes.byref(size),
                ctypes.byref(nesting_depth),
                ctypes.byref(info),
                ctypes.byref(count),
            )
            if kr != 0:
                break

            total_found += 1
            if skipped < offset:
                skipped += 1
                address.value += size.value
                continue

            if len(regions) < limit:
                tag = info.user_tag
                tag_name = VM_USER_TAGS.get(tag, f"TAG_{tag}")
                prot_str = format_protection(info.protection)
                max_prot_str = format_protection(info.max_protection)

                regions.append({
                    "start_address": f"0x{address.value:x}",
                    "end_address": f"0x{address.value + size.value:x}",
                    "size_bytes": size.value,
                    "size_human": format_size(size.value),
                    "protection": prot_str,
                    "max_protection": max_prot_str,
                    "user_tag": tag_name,
                    "is_submap": bool(info.is_submap),
                })

            address.value += size.value
            if len(regions) >= limit:
                break

        return {
            "pid": pid,
            "offset": offset,
            "limit": limit,
            "count": len(regions),
            "total_scanned": total_found,
            "regions": regions,
        }

    def search_memory(
        self,
        pid: int,
        value: Any,
        data_type: str = "int32",
        max_results: int = 50,
        region_type: str = "writable",
    ) -> dict[str, Any]:
        """Scan memory regions of target process for specific value or byte patterns."""
        if max_results <= 0:
            raise ValueError("max_results must be greater than 0")

        dtype = data_type.lower()
        try:
            if dtype == "int32":
                needle = struct.pack("<i", int(value))
            elif dtype == "int64":
                needle = struct.pack("<q", int(value))
            elif dtype == "float":
                needle = struct.pack("<f", float(value))
            elif dtype == "double":
                needle = struct.pack("<d", float(value))
            elif dtype == "string":
                needle = str(value).encode("utf-8")
            elif dtype == "bytes":
                clean_hex = str(value).strip().replace("0x", "").replace(" ", "")
                needle = bytes.fromhex(clean_hex)
            else:
                raise ValueError(
                    f"Unsupported data_type: {data_type}. Choose from 'int32', 'int64', 'float', 'double', 'bytes', 'string'."
                )
        except Exception as e:
            raise ValueError(f"Failed to encode value '{value}' as '{data_type}': {e}")

        needle_len = len(needle)
        if needle_len == 0:
            raise ValueError("Search target cannot be empty")

        task = self.get_task_port(pid)
        if not _mach_vm_region_recurse or not _mach_vm_read_overwrite:
            raise RuntimeError("Mach VM kernel APIs unavailable")

        address = ctypes.c_uint64(0)
        size = ctypes.c_uint64(0)
        nesting_depth = ctypes.c_uint32(0)
        info = vm_region_submap_info_64()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)

        matches: list[dict[str, Any]] = []
        max_chunk_size = 1024 * 1024  # 1 MB dynamic chunk
        buf = ctypes.create_string_buffer(max_chunk_size)
        buf_addr = ctypes.addressof(buf)
        out_size = ctypes.c_size_t(0)

        reg_mode = region_type.lower()

        while True:
            kr = _mach_vm_region_recurse(
                task,
                ctypes.byref(address),
                ctypes.byref(size),
                ctypes.byref(nesting_depth),
                ctypes.byref(info),
                ctypes.byref(count),
            )
            if kr != 0:
                break

            is_candidate = False
            if not info.is_submap:
                if reg_mode == "writable" and (info.protection & 0x03) == 0x03:
                    is_candidate = True
                elif reg_mode == "all" and (info.protection & 0x01) == 0x01:
                    is_candidate = True

            if is_candidate and size.value > 0:
                reg_start = address.value
                reg_len = size.value
                reg_end = reg_start + reg_len
                curr = reg_start
                tag_name = VM_USER_TAGS.get(info.user_tag, f"TAG_{info.user_tag}")
                prot_str = format_protection(info.protection)

                while curr < reg_end:
                    rem = reg_end - curr
                    to_read = min(max_chunk_size, rem)
                    if to_read > PAGE_SIZE:
                        to_read = (to_read // PAGE_SIZE) * PAGE_SIZE

                    kr_read = _mach_vm_read_overwrite(
                        task,
                        ctypes.c_uint64(curr),
                        ctypes.c_uint64(to_read),
                        ctypes.c_uint64(buf_addr),
                        ctypes.byref(out_size),
                    )
                    if kr_read == 0 and out_size.value >= needle_len:
                        data = buf.raw[: out_size.value]
                        pos = 0
                        while pos <= len(data) - needle_len:
                            idx = data.find(needle, pos)
                            if idx == -1:
                                break
                            match_addr = curr + idx
                            matches.append({
                                "address": f"0x{match_addr:x}",
                                "offset": idx,
                                "region_base": f"0x{reg_start:x}",
                                "region_size": reg_len,
                                "protection": prot_str,
                                "user_tag": tag_name,
                            })
                            if len(matches) >= max_results:
                                break
                            pos = idx + 1

                        if len(matches) >= max_results:
                            break
                        step = to_read if to_read <= needle_len else (to_read - (needle_len - 1))
                        curr += step
                    elif to_read > PAGE_SIZE:
                        # Fallback to page-by-page traversal (16 KB) for unmapped/guard pages within this chunk
                        page_curr = curr
                        chunk_end = curr + to_read
                        while page_curr < chunk_end:
                            p_read = min(PAGE_SIZE, chunk_end - page_curr)
                            kr_page = _mach_vm_read_overwrite(
                                task,
                                ctypes.c_uint64(page_curr),
                                ctypes.c_uint64(p_read),
                                ctypes.c_uint64(buf_addr),
                                ctypes.byref(out_size),
                            )
                            if kr_page == 0 and out_size.value >= needle_len:
                                p_data = buf.raw[: out_size.value]
                                pos = 0
                                while pos <= len(p_data) - needle_len:
                                    idx = p_data.find(needle, pos)
                                    if idx == -1:
                                        break
                                    match_addr = page_curr + idx
                                    matches.append({
                                        "address": f"0x{match_addr:x}",
                                        "offset": idx,
                                        "region_base": f"0x{reg_start:x}",
                                        "region_size": reg_len,
                                        "protection": prot_str,
                                        "user_tag": tag_name,
                                    })
                                    if len(matches) >= max_results:
                                        break
                                    pos = idx + 1

                            if len(matches) >= max_results:
                                break
                            page_curr += p_read

                        if len(matches) >= max_results:
                            break
                        curr += to_read
                    else:
                        curr += PAGE_SIZE

            if len(matches) >= max_results:
                break

            address.value += size.value

        return {
            "pid": pid,
            "data_type": data_type,
            "search_value": value,
            "needle_hex": needle.hex(),
            "region_type": region_type,
            "total_found": len(matches),
            "matches": matches,
        }

    def read_memory(
        self, pid: int, address_str: str, size: int, fmt: str = "hex"
    ) -> dict[str, Any]:
        """Read structured memory from process address."""
        if size <= 0:
            raise ValueError("Read size must be greater than 0")
        if size > 10 * 1024 * 1024:
            raise ValueError("Read size exceeds 10 MB safety ceiling")

        addr = strip_pac(parse_address(address_str))
        task = self.get_task_port(pid)

        # First attempt via LLDB if active
        if self.lldb_process and self.lldb_process.IsValid() and self.active_pid == pid:
            err = lldb.SBError()
            buf = bytearray(size)
            bytes_read = self.lldb_process.ReadMemory(addr, buf, err)
            if err.Success() and bytes_read == size:
                raw_bytes = bytes(buf)
                parsed = format_bytes(raw_bytes, fmt)
                return {
                    "pid": pid,
                    "address": f"0x{addr:x}",
                    "size": size,
                    "format": fmt,
                    "value": parsed,
                    "hex": raw_bytes.hex(),
                }

        # Native Mach VM read
        if not _mach_vm_read_overwrite:
            raise RuntimeError("mach_vm_read_overwrite API not available")

        out_buf = ctypes.create_string_buffer(size)
        out_size = ctypes.c_size_t(0)

        kr = _mach_vm_read_overwrite(
            task,
            ctypes.c_uint64(addr),
            ctypes.c_uint64(size),
            ctypes.addressof(out_buf),
            ctypes.byref(out_size),
        )

        if kr != 0:
            err_desc = mach_error_string(kr)
            raise ValueError(f"Read failed at 0x{addr:x} (size {size}): {err_desc}")

        raw_bytes = bytes(out_buf.raw[: out_size.value])
        parsed = format_bytes(raw_bytes, fmt)

        return {
            "pid": pid,
            "address": f"0x{addr:x}",
            "size": out_size.value,
            "format": fmt,
            "value": parsed,
            "hex": raw_bytes.hex(),
        }

    def write_memory(
        self, pid: int, address_str: str, bytes_hex: str
    ) -> dict[str, Any]:
        """Modify memory state at address (with automatic page protection override if needed)."""
        addr = strip_pac(parse_address(address_str))
        clean_hex = bytes_hex.strip().replace(" ", "").replace("0x", "")
        try:
            payload = bytes.fromhex(clean_hex)
        except ValueError as e:
            raise ValueError(f"Invalid hex string payload: {e}")

        payload_len = len(payload)
        if payload_len == 0:
            raise ValueError("Payload cannot be empty")

        task = self.get_task_port(pid)

        # Attempt LLDB write if available
        if self.lldb_process and self.lldb_process.IsValid() and self.active_pid == pid:
            err = lldb.SBError()
            written = self.lldb_process.WriteMemory(addr, payload, err)
            if err.Success() and written == payload_len:
                return {
                    "status": "success",
                    "pid": pid,
                    "address": f"0x{addr:x}",
                    "bytes_written": written,
                    "hex": payload.hex(),
                }

        # Native Mach VM write
        if not _mach_vm_write:
            raise RuntimeError("mach_vm_write API not available")

        kr = _mach_vm_write(task, addr, payload, payload_len)

        # If protection failure, attempt temporary page protection escalation (VM_PROT_READ | VM_PROT_WRITE)
        if kr == 2 and _mach_vm_protect:
            page_size = 16384  # Standard ARM64 Apple Silicon page size
            page_addr = addr & ~(page_size - 1)
            protect_size = ((addr + payload_len - page_addr + page_size - 1) // page_size) * page_size
            _mach_vm_protect(task, page_addr, protect_size, 0, 0x01 | 0x02)  # RW
            kr = _mach_vm_write(task, addr, payload, payload_len)

        if kr != 0:
            err_desc = mach_error_string(kr)
            raise ValueError(f"Write failed at 0x{addr:x} (len {payload_len}): {err_desc}")

        return {
            "status": "success",
            "pid": pid,
            "address": f"0x{addr:x}",
            "bytes_written": payload_len,
            "hex": payload.hex(),
        }

    def _set_watchpoint_external_bridge(
        self, pid: int, address: int, size: int, watch_type: str
    ) -> dict[str, Any]:
        """Headless LLDB external bridge via system Python (/usr/bin/python3)."""
        read = watch_type in ("read", "read_write")
        write = watch_type in ("write", "read_write")

        # Do not allow child process to attach to parent or self to prevent circular debugger SIGBUS (-10)
        if pid == os.getpid():
            raise RuntimeError(
                f"Cannot attach debugger bridge to self PID {pid} (causes kernel debug lock / SIGBUS). Attach to a separate target process."
            )

        script = f"""
import sys, json, os
sys.path.append("/Applications/Xcode.app/Contents/SharedFrameworks/LLDB.framework/Resources/Python")
try:
    import lldb
    debugger = lldb.SBDebugger.Create()
    debugger.SetAsync(False)
    target = debugger.CreateTarget("")
    if not target or not target.IsValid():
        print(json.dumps({{"status": "error", "error": "Failed to create LLDB target"}}))
        sys.exit(0)
    
    err = lldb.SBError()
    pid = {pid}
    if pid > 0:
        listener = lldb.SBListener("bridge.listener")
        process = target.AttachToProcessWithID(listener, pid, err)
        if not err.Success():
            print(json.dumps({{"status": "error", "error": f"Attach failed: {{err.GetCString()}}"}}))
            sys.exit(0)

    wp_err = lldb.SBError()
    wp = target.WatchAddress({address}, {size}, {read}, {write}, wp_err)
    if not wp_err.Success() or not wp or not wp.IsValid():
        err_msg = wp_err.GetCString() if not wp_err.Success() else "Failed to create watchpoint"
        print(json.dumps({{"status": "error", "error": err_msg}}))
    else:
        print(json.dumps({{
            "status": "active",
            "watchpoint_id": wp.GetID(),
            "address": hex({address}),
            "size": {size},
            "watch_type": "{watch_type}",
            "hardware": True,
            "bridge": "lldb_system_python"
        }}))
    lldb.SBDebugger.Destroy(debugger)
except Exception as ex:
    print(json.dumps({{"status": "error", "error": str(ex)}}))
"""
        try:
            res = subprocess.run(
                ["/usr/bin/python3", "-c", script],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"LLDB bridge subprocess exited with code {res.returncode}: {res.stderr.strip()}"
                )

            out = res.stdout.strip()
            data = json.loads(out)
            if data.get("status") == "error":
                raise RuntimeError(f"LLDB watchpoint error: {data.get('error')}")
            return data
        except json.JSONDecodeError:
            raise RuntimeError(
                f"Invalid JSON response from LLDB bridge: {res.stdout.strip()}"
            )
        except Exception as e:
            if isinstance(e, RuntimeError):
                raise
            raise RuntimeError(f"Failed to execute LLDB external bridge: {e}")

    def set_watchpoint(
        self, address_str: str, size: int, watch_type: str = "write"
    ) -> dict[str, Any]:
        """Set ARM64 hardware watchpoint via LLDB (in-process or resilient bridge)."""
        if size not in (1, 2, 4, 8):
            raise ValueError(f"Watchpoint size must be 1, 2, 4, or 8 bytes, got {size}")

        watch_type = watch_type.lower()
        if watch_type not in ("read", "write", "read_write"):
            raise ValueError("watch_type must be 'read', 'write', or 'read_write'")

        addr = strip_pac(parse_address(address_str))

        # 1. If in-process LLDB is attached and valid
        if (
            self.lldb_target
            and self.lldb_target.IsValid()
            and self.lldb_process
            and self.lldb_process.IsValid()
        ):
            read = watch_type in ("read", "read_write")
            write = watch_type in ("write", "read_write")
            err = lldb.SBError()
            wp = self.lldb_target.WatchAddress(addr, size, read, write, err)
            if err.Success() and wp and wp.IsValid():
                return {
                    "status": "active",
                    "watchpoint_id": wp.GetID(),
                    "address": f"0x{addr:x}",
                    "size": size,
                    "watch_type": watch_type,
                    "hardware": True,
                    "bridge": "in_process_lldb",
                }

        # 2. Resilient external LLDB bridge
        pid = self.active_pid or os.getpid()
        return self._set_watchpoint_external_bridge(pid, addr, size, watch_type)
