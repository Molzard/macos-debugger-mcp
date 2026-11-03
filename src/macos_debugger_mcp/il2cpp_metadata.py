"""IL2CPP Global Metadata Parser supporting Unity versions 24 through 39.

Extracts TypeDefinitions, FieldDefinitions, string literal pools, and image definitions.
Produces indexed caches mapping class tokens and names to fields and types.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FieldInfoDef:
    name: str
    type_index: int
    token: uint32 = 0
    offset: int = 0
    type_name: str = "Unknown"


@dataclass
class TypeInfoDef:
    name: str
    namespace: str
    token: int
    type_index: int
    image_name: str
    field_start: int
    field_count: int
    method_start: int
    method_count: int
    parent_index: int
    byval_type_index: int
    fields: List[FieldInfoDef] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        if self.namespace:
            return f"{self.namespace}.{self.name}"
        return self.name


@dataclass
class ImageInfoDef:
    name: str
    assembly_index: int
    type_start: int
    type_count: int


# IL2CPP Type Enums (System types)
IL2CPP_TYPE_MAP = {
    0x01: "void",
    0x02: "bool",
    0x03: "char",
    0x04: "sbyte",
    0x05: "byte",
    0x06: "int16",
    0x07: "uint16",
    0x08: "int32",
    0x09: "uint32",
    0x0A: "int64",
    0x0B: "uint64",
    0x0C: "single",
    0x0D: "double",
    0x0E: "string",
    0x0F: "ptr",
    0x10: "byref",
    0x11: "valuetype",
    0x12: "class",
    0x13: "var",
    0x14: "array",
    0x15: "genericinst",
    0x16: "typedbyref",
    0x18: "intptr",
    0x19: "uintptr",
    0x1C: "object",
    0x1D: "szarray",
    0x1E: "mvar",
}

TYPE_SIZES = {
    "bool": 1,
    "byte": 1,
    "sbyte": 1,
    "int16": 2,
    "uint16": 2,
    "char": 2,
    "int32": 4,
    "uint32": 4,
    "single": 4,
    "int64": 8,
    "uint64": 8,
    "double": 8,
    "ptr": 8,
    "intptr": 8,
    "uintptr": 8,
    "string": 8,
    "class": 8,
    "object": 8,
    "szarray": 8,
    "genericinst": 8,
    "valuetype": 8,
    "Unknown": 8,
}


class Il2CppMetadataParser:
    """Parses Unity IL2CPP global-metadata.dat files across versions 24..39."""

    SANITY_MAGIC = 0xFAB11BAF

    def __init__(self, file_path: str):
        self.file_path = os.path.expanduser(file_path)
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"Metadata file not found: {self.file_path}")

        with open(self.file_path, "rb") as f:
            self.data = f.read()

        if len(self.data) < 256:
            raise ValueError("Metadata file is too small to contain a valid header.")

        self.magic, self.version = struct.unpack_from("<II", self.data, 0)
        if self.magic != self.SANITY_MAGIC:
            raise ValueError(f"Invalid IL2CPP metadata magic: {hex(self.magic)} (expected {hex(self.SANITY_MAGIC)})")

        self.sections: Dict[str, Tuple[int, int, int]] = {}
        self._parse_header()

        # Cache lookups
        self.images: List[ImageInfoDef] = []
        self.type_definitions: List[TypeInfoDef] = []
        self.types_by_name: Dict[str, TypeInfoDef] = {}
        self.types_by_token: Dict[int, TypeInfoDef] = {}
        self.types_by_full_name_lower: Dict[str, TypeInfoDef] = {}

        self._parse_images()
        self._parse_type_definitions()
        self._parse_fields()

    def _parse_header(self) -> None:
        """Parse header sections based on version format."""
        section_names = [
            "stringLiteral",
            "stringLiteralData",
            "string",
            "events",
            "properties",
            "methods",
            "parameterDefaultValues",
            "fieldDefaultValues",
            "fieldAndParameterDefaultValueData",
            "fieldMarshaledSizes",
            "parameters",
            "fields",
            "genericParameters",
            "genericParameterConstraints",
            "genericContainers",
            "nestedTypes",
            "interfaces",
            "vtableMethods",
            "interfaceOffsets",
            "typeDefinitions",
            "images",
            "assemblies",
            "fieldRefs",
            "referencedAssemblies",
            "attributeData",
            "attributeDataRange",
            "unresolvedVirtualCallParameterTypes",
            "unresolvedVirtualCallParameterRanges",
            "windowsRuntimeTypeNames",
            "windowsRuntimeStrings",
            "exportedTypeDefinitions",
        ]

        if self.version >= 38:
            # Modern header (v38..v39): triplet of (offset: uint32, size: uint32, count: uint32) = 12 bytes
            pos = 8
            for name in section_names:
                if pos + 12 <= len(self.data):
                    off, sz, cnt = struct.unpack_from("<III", self.data, pos)
                    self.sections[name] = (off, sz, cnt)
                    pos += 12
        else:
            # Legacy header (v24..v31): pairs of (offset: uint32, size: uint32) = 8 bytes
            pos = 8
            for name in section_names:
                if pos + 8 <= len(self.data):
                    off, sz = struct.unpack_from("<II", self.data, pos)
                    self.sections[name] = (off, sz, 0)
                    pos += 8

    def get_string(self, index: int) -> str:
        """Resolve a null-terminated string from metadata string table."""
        if "string" not in self.sections:
            return ""
        str_offset, str_size, _ = self.sections["string"]
        if index >= str_size:
            return ""
        start = str_offset + index
        end = self.data.find(b"\x00", start)
        if end == -1:
            end = min(start + 256, len(self.data))
        return self.data[start:end].decode("utf-8", errors="replace")

    def get_string_literal(self, index: int) -> str:
        """Resolve a managed string literal by index."""
        if "stringLiteral" not in self.sections or "stringLiteralData" not in self.sections:
            return ""
        lit_offset, lit_size, _ = self.sections["stringLiteral"]
        data_offset, data_size, _ = self.sections["stringLiteralData"]

        # In modern v38+, stringLiteral is an array of 4-byte offsets into stringLiteralData
        record_size = 4 if self.version >= 35 else 8
        rec_pos = lit_offset + index * record_size
        if rec_pos + record_size > lit_offset + lit_size:
            return ""

        if record_size == 4:
            offset_in_data = struct.unpack_from("<I", self.data, rec_pos)[0]
            start = data_offset + offset_in_data
            end = self.data.find(b"\x00", start)
            if end == -1:
                end = min(start + 512, len(self.data))
            return self.data[start:end].decode("utf-8", errors="replace")
        else:
            length, data_idx = struct.unpack_from("<II", self.data, rec_pos)
            start = data_offset + data_idx
            return self.data[start:start + length].decode("utf-8", errors="replace")

    def _parse_images(self) -> None:
        """Parse image definitions (assembly containers)."""
        if "images" not in self.sections:
            return
        off, sz, cnt = self.sections["images"]
        # In v38/v39 record size is 36 bytes, legacy is 40 bytes
        rec_size = 36 if self.version >= 38 else 40
        count = cnt if cnt > 0 else sz // rec_size

        for i in range(count):
            entry_off = off + i * rec_size
            if entry_off + 16 > len(self.data):
                break
            name_idx, asm_idx, type_start, type_count = struct.unpack_from("<IIII", self.data, entry_off)
            img_name = self.get_string(name_idx)
            self.images.append(ImageInfoDef(
                name=img_name,
                assembly_index=asm_idx,
                type_start=type_start,
                type_count=type_count,
            ))

    def _parse_type_definitions(self) -> None:
        """Parse all TypeDefinitions from metadata."""
        if "typeDefinitions" not in self.sections:
            return
        off, sz, cnt = self.sections["typeDefinitions"]
        # v38/v39 record size is 82 bytes, v24-v31 is 88 bytes, v24.1 is 96 bytes
        rec_size = 82 if self.version >= 38 else 88
        count = cnt if cnt > 0 else sz // rec_size

        # Associate types with images
        image_idx = 0
        current_image = self.images[0].name if self.images else "Unknown"

        for i in range(count):
            entry_off = off + i * rec_size
            if entry_off + 40 > len(self.data):
                break

            name_idx, ns_idx = struct.unpack_from("<II", self.data, entry_off)
            name = self.get_string(name_idx)
            namespace = self.get_string(ns_idx)

            if rec_size == 82:
                # Modern layout (v38/v39)
                byval_idx, declaring_idx, parent_idx, generic_container_idx = struct.unpack_from("<HHHH", self.data, entry_off + 8)
                flags = struct.unpack_from("<I", self.data, entry_off + 20)[0]
                field_start = struct.unpack_from("<I", self.data, entry_off + 26)[0]
                method_start = struct.unpack_from("<I", self.data, entry_off + 30)[0]
                method_count, prop_count, field_count, event_count, nested_count, vtable_count, iface_count, iface_off_count = struct.unpack_from("<8H", self.data, entry_off + 58)
                bitfield, token = struct.unpack_from("<II", self.data, entry_off + 74)
            else:
                # Standard v24-v31 layout
                byval_idx = struct.unpack_from("<i", self.data, entry_off + 8)[0]
                parent_idx = struct.unpack_from("<i", self.data, entry_off + 16)[0]
                field_start = struct.unpack_from("<i", self.data, entry_off + 28)[0]
                method_start = struct.unpack_from("<i", self.data, entry_off + 32)[0]
                method_count, _, field_count = struct.unpack_from("<HHH", self.data, entry_off + 60)
                token = struct.unpack_from("<I", self.data, entry_off + 84)[0]

            # Find matching image
            while image_idx + 1 < len(self.images) and i >= self.images[image_idx + 1].type_start:
                image_idx += 1
                current_image = self.images[image_idx].name

            type_info = TypeInfoDef(
                name=name,
                namespace=namespace,
                token=token,
                type_index=i,
                image_name=current_image,
                field_start=field_start,
                field_count=field_count,
                method_start=method_start,
                method_count=method_count,
                parent_index=parent_idx,
                byval_type_index=byval_idx,
            )

            self.type_definitions.append(type_info)
            if name:
                self.types_by_name[name] = type_info
                self.types_by_name[type_info.full_name] = type_info
                self.types_by_full_name_lower[type_info.full_name.lower()] = type_info
                self.types_by_full_name_lower[name.lower()] = type_info
            if token:
                self.types_by_token[token] = type_info

    def _parse_fields(self) -> None:
        """Parse FieldDefinitions and associate them with TypeDefinitions."""
        if "fields" not in self.sections:
            return
        off, sz, cnt = self.sections["fields"]
        rec_size = 12  # nameIndex(4), typeIndex(4), token(4)
        total_fields = cnt if cnt > 0 else sz // rec_size

        for tdef in self.type_definitions:
            if tdef.field_count <= 0 or tdef.field_start < 0:
                continue

            current_offset = 0x10  # Standard IL2CPP ARM64 instance field start (0x0 klass*, 0x8 monitor*)

            for f_idx in range(tdef.field_start, min(tdef.field_start + tdef.field_count, total_fields)):
                f_off = off + f_idx * rec_size
                if f_off + 12 > len(self.data):
                    break

                fname_idx, ftype_idx, ftoken = struct.unpack_from("<IiI", self.data, f_off)
                field_name = self.get_string(fname_idx)

                # Heuristic type size and alignment
                type_name = "object"
                field_size = 8
                # Align to 8 bytes for pointer/reference
                if current_offset % field_size != 0:
                    current_offset += field_size - (current_offset % field_size)

                field_def = FieldInfoDef(
                    name=field_name,
                    type_index=ftype_idx,
                    token=ftoken,
                    offset=current_offset,
                    type_name=type_name,
                )
                tdef.fields.append(field_def)
                current_offset += field_size

    def find_type(self, identifier: str) -> Optional[TypeInfoDef]:
        """Find a type definition by simple name or full namespace.name."""
        if identifier in self.types_by_name:
            return self.types_by_name[identifier]
        lowered = identifier.lower()
        if lowered in self.types_by_full_name_lower:
            return self.types_by_full_name_lower[lowered]
        return None
