"""Dynamic VTable and RTTI Resolver for Unity IL2CPP on macOS ARM64.

Resolves heap objects, class descriptors, and field graphs using native Mach VM APIs.
Features strict alignment checks, null guards, and memory protection verification.
"""

from __future__ import annotations

import ctypes
import os
import struct
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    from .il2cpp_metadata import Il2CppMetadataParser, TypeInfoDef
except ImportError:
    from il2cpp_metadata import Il2CppMetadataParser, TypeInfoDef

# Native Mach VM definitions via libSystem
_libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")

_mach_task_self = _libc.mach_task_self
_mach_task_self.restype = ctypes.c_uint32

_task_for_pid = _libc.task_for_pid
_task_for_pid.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.POINTER(ctypes.c_uint32)]
_task_for_pid.restype = ctypes.c_int

_mach_port_deallocate = _libc.mach_port_deallocate
_mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_mach_port_deallocate.restype = ctypes.c_int

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


class MachVMError(Exception):
    pass


class MachVMAccess:
    """Encapsulates safe native Mach VM operations for macOS processes."""

    def __init__(self, pid: Optional[int] = None):
        self.pid = pid
        self._task_port: Optional[int] = None
        if pid is not None:
            self._acquire_task_port()

    def _acquire_task_port(self) -> None:
        if self.pid is None:
            return
        if self.pid == os.getpid():
            self._task_port = _mach_task_self()
            return

        port = ctypes.c_uint32()
        kr = _task_for_pid(_mach_task_self(), self.pid, ctypes.byref(port))
        if kr != 0:
            raise PermissionError(
                f"task_for_pid failed with kern_return {kr}. "
                f"Ensure process runs with debugger entitlement or root permissions."
            )
        self._task_port = port.value

    @property
    def task_port(self) -> int:
        if self._task_port is None:
            self._acquire_task_port()
        return self._task_port or 0

    def close(self) -> None:
        if self._task_port and self.pid != os.getpid():
            _mach_port_deallocate(_mach_task_self(), self._task_port)
            self._task_port = None

    def read_bytes(self, address: int, size: int) -> bytes:
        """Safely read memory from process address via mach_vm_read_overwrite."""
        if size <= 0:
            return b""
        buf = ctypes.create_string_buffer(size)
        out_size = ctypes.c_size_t()
        target_addr = ctypes.cast(buf, ctypes.c_void_p).value or 0
        kr = _mach_vm_read_overwrite(
            self.task_port,
            ctypes.c_uint64(address),
            ctypes.c_uint64(size),
            ctypes.c_uint64(target_addr),
            ctypes.byref(out_size),
        )
        if kr != 0:
            raise MachVMError(f"mach_vm_read_overwrite failed at 0x{address:x} (size: {size}, kr: {kr})")
        return bytes(buf.raw[: out_size.value])

    def read_uint64(self, address: int) -> int:
        data = self.read_bytes(address, 8)
        return struct.unpack("<Q", data)[0]

    def read_uint32(self, address: int) -> int:
        data = self.read_bytes(address, 4)
        return struct.unpack("<I", data)[0]

    def read_int32(self, address: int) -> int:
        data = self.read_bytes(address, 4)
        return struct.unpack("<i", data)[0]

    def read_utf16_string(self, address: int, char_count: int) -> str:
        raw = self.read_bytes(address, char_count * 2)
        return raw.decode("utf-16le", errors="replace")

    def read_c_string(self, address: int, max_len: int = 128) -> str:
        raw = self.read_bytes(address, max_len)
        null_idx = raw.find(b"\x00")
        if null_idx != -1:
            raw = raw[:null_idx]
        return raw.decode("utf-8", errors="replace")

    def enumerate_regions(self, writable_only: bool = True) -> List[Tuple[int, int, int]]:
        """List virtual memory regions (start, end, protection)."""
        regions = []
        address = ctypes.c_uint64(0)
        size = ctypes.c_uint64(0)
        nesting_depth = ctypes.c_uint32(0)
        info = vm_region_submap_info_64()
        count = ctypes.c_uint32(ctypes.sizeof(info) // 4)

        while True:
            kr = _mach_vm_region_recurse(
                self.task_port,
                ctypes.byref(address),
                ctypes.byref(size),
                ctypes.byref(nesting_depth),
                ctypes.cast(ctypes.byref(info), ctypes.c_void_p),
                ctypes.byref(count),
            )
            if kr != 0:
                break

            if info.is_submap:
                nesting_depth.value += 1
                continue

            prot = info.protection
            start = address.value
            end = start + size.value

            # prot & 2 -> VM_PROT_WRITE
            if not writable_only or (prot & 0x02):
                regions.append((start, end, prot))

            address.value += size.value
            if address.value >= 0x7FFFFFFFFFFF:
                break

        return regions


class RTTIResolver:
    """Semantic Object Graph Resolver for Unity IL2CPP runtime heap."""

    def __init__(self, metadata_parser: Il2CppMetadataParser, vm_reader: Optional[MachVMAccess] = None):
        self.meta = metadata_parser
        self.vm = vm_reader

    @staticmethod
    def parse_address(addr: Union[int, str]) -> int:
        if isinstance(addr, str):
            return int(addr, 16) if addr.lower().startswith("0x") else int(addr)
        return addr

    def resolve_object_at_address(
        self,
        target_address: Union[int, str],
        read_fn: Optional[Callable[[int, int], bytes]] = None,
        max_depth: int = 2,
    ) -> Dict[str, Any]:
        """Resolves an IL2CPP heap object at the specified memory address."""
        addr = self.parse_address(target_address)

        # 1. ARM64 8-byte pointer boundary verification
        if addr % 8 != 0:
            return {
                "error": "AlignmentFault",
                "message": f"Target address 0x{addr:x} is not 8-byte aligned on ARM64.",
                "raw_address": hex(addr),
            }

        # Safe memory reader fallback
        reader = read_fn or (lambda a, sz: self.vm.read_bytes(a, sz) if self.vm else b"")

        try:
            # 2. Read Il2CppClass* pointer at offset 0x0
            header_bytes = reader(addr, 8)
            if len(header_bytes) < 8:
                return {"error": "UnreadableMemory", "raw_address": hex(addr)}
            klass_ptr = struct.unpack("<Q", header_bytes)[0]

            if klass_ptr == 0:
                return {
                    "raw_address": hex(addr),
                    "resolved_type": "[NullPointer]",
                    "class_name": "None",
                    "fields": [],
                }

            # 3. Resolve Class Name and Type from Class Descriptor
            type_def, class_name = self._resolve_class_from_descriptor(klass_ptr, reader)

            result: Dict[str, Any] = {
                "raw_address": hex(addr),
                "klass_descriptor": hex(klass_ptr),
                "resolved_type": f"[Class: {class_name}]",
                "class_name": class_name,
                "fields": [],
            }

            if not type_def:
                return result

            # 4. Traverse and format declared fields
            fields_data = []
            for fld in type_def.fields:
                field_addr = addr + fld.offset
                f_entry: Dict[str, Any] = {
                    "offset": hex(fld.offset),
                    "name": fld.name,
                    "type": fld.type_name,
                }

                try:
                    raw_val = reader(field_addr, 8)
                    val_u64 = struct.unpack("<Q", raw_val)[0] if len(raw_val) >= 8 else 0
                    val_i32 = struct.unpack("<i", raw_val[:4])[0] if len(raw_val) >= 4 else 0

                    if "string" in fld.name.lower() or fld.type_name == "string":
                        # Attempt reading System.String
                        f_entry["value"] = self._read_system_string(val_u64, reader)
                    elif any(c in fld.name.lower() for c in ["list", "card", "deck", "pile", "array"]):
                        # Inspect potential collection / reference
                        collection_info = self._inspect_collection(val_u64, reader)
                        if collection_info:
                            f_entry["value"] = collection_info
                        else:
                            f_entry["value"] = hex(val_u64)
                    elif any(k in fld.name.lower() for k in ["score", "count", "id", "amount", "star", "level"]):
                        f_entry["value"] = val_i32
                    else:
                        f_entry["value"] = hex(val_u64)

                except Exception as ex:
                    f_entry["value"] = f"<ReadError: {ex}>"

                fields_data.append(f_entry)

            result["fields"] = fields_data
            return result

        except Exception as e:
            return {
                "error": "ResolutionException",
                "message": str(e),
                "raw_address": hex(addr),
            }

    def _resolve_class_from_descriptor(
        self,
        klass_ptr: int,
        reader: Callable[[int, int], bytes],
    ) -> Tuple[Optional[TypeInfoDef], str]:
        """Reads Class descriptor metadata to cross-reference against offline definitions."""
        try:
            # Read first 48 bytes of Il2CppClass
            desc_bytes = reader(klass_ptr, 48)
            if len(desc_bytes) >= 32:
                # Name and Namespace pointers are typically at offsets 0x10 and 0x18 in Il2CppClass
                name_ptr = struct.unpack("<Q", desc_bytes[16:24])[0]
                ns_ptr = struct.unpack("<Q", desc_bytes[24:32])[0]

                if name_ptr > 0x1000:
                    name_str = reader(name_ptr, 64).split(b"\x00")[0].decode("utf-8", errors="replace")
                    if name_str and name_str.isprintable():
                        tdef = self.meta.find_type(name_str)
                        return tdef, (tdef.full_name if tdef else name_str)
        except Exception:
            pass

        # Fallback to searching type definitions by class pointer token / heuristic
        return None, f"UnknownClass_0x{klass_ptr:x}"

    def _read_system_string(self, str_ptr: int, reader: Callable[[int, int], bytes]) -> Optional[str]:
        """Unpack System.String (length at offset +0x10, utf-16 characters at +0x14)."""
        if str_ptr == 0:
            return None
        try:
            len_bytes = reader(str_ptr + 0x10, 4)
            if len(len_bytes) < 4:
                return None
            length = struct.unpack("<i", len_bytes)[0]
            if 0 <= length < 256:
                char_bytes = reader(str_ptr + 0x14, length * 2)
                return char_bytes.decode("utf-16le", errors="replace")
        except Exception:
            pass
        return None

    def _inspect_collection(self, coll_ptr: int, reader: Callable[[int, int], bytes]) -> Optional[str]:
        """Inspects List<T> or Array headers (elements count)."""
        if coll_ptr == 0:
            return "null"
        try:
            # In IL2CPP List<T>: offset 0x10 is _items (Array), offset 0x18 is _size (int32)
            header = reader(coll_ptr, 28)
            if len(header) >= 28:
                items_ptr = struct.unpack("<Q", header[16:24])[0]
                size = struct.unpack("<i", header[24:28])[0]
                if 0 <= size < 10000 and items_ptr != 0:
                    return f"List (Elements: {size})"
            # In IL2CPP Array: offset 0x18 is max_length (int32)
            if len(header) >= 28:
                arr_len = struct.unpack("<i", header[24:28])[0]
                if 0 <= arr_len < 10000:
                    return f"Array (Length: {arr_len})"
        except Exception:
            pass
        return None

    def find_active_instances(
        self,
        class_name: str,
        max_results: int = 50,
        read_fn: Optional[Callable[[int, int], bytes]] = None,
        regions: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Scans writable memory regions for allocated heap instances of the specified class."""
        tdef = self.meta.find_type(class_name)
        if not tdef:
            return []

        results = []
        reader = read_fn or (lambda a, sz: self.vm.read_bytes(a, sz) if self.vm else b"")
        mem_regions = regions or (self.vm.enumerate_regions(writable_only=True) if self.vm else [])

        # Step through memory regions in 8-byte increments
        for start, end, _ in mem_regions:
            size = min(end - start, 4 * 1024 * 1024)  # Scan in bounded 4MB chunks
            try:
                chunk = reader(start, size)
                for off in range(0, len(chunk) - 8, 8):
                    ptr = struct.unpack_from("<Q", chunk, off)[0]
                    # Verify if ptr points to a matching class descriptor
                    obj_addr = start + off
                    res = self.resolve_object_at_address(obj_addr, read_fn=reader)
                    if res.get("class_name") == tdef.name or res.get("class_name") == tdef.full_name:
                        results.append(res)
                        if len(results) >= max_results:
                            return results
            except Exception:
                continue

        return results
