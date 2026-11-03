"""Test Harness for Unity IL2CPP Semantic Runtime Profiler."""

import ctypes
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from macos_debugger_mcp.il2cpp_metadata import Il2CppMetadataParser, TypeInfoDef, FieldInfoDef
    from macos_debugger_mcp.rtti_resolver import RTTIResolver, MachVMAccess
except ImportError:
    from il2cpp_metadata import Il2CppMetadataParser, TypeInfoDef, FieldInfoDef
    from rtti_resolver import RTTIResolver, MachVMAccess


class TestSemanticResolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata_path = os.path.expanduser(
            "~/Library/Containers/io.playcover.PlayCover/Applications/com.rovio.baba.app/Data/Managed/Metadata/global-metadata.dat"
        )
        if os.path.isfile(cls.metadata_path):
            cls.parser = Il2CppMetadataParser(cls.metadata_path)
        else:
            cls.parser = None

    def test_01_metadata_parser(self):
        """Verify global-metadata.dat header and TypeDefinitions extraction."""
        if not self.parser:
            self.skipTest("global-metadata.dat not found on host")

        self.assertEqual(self.parser.magic, 0xFAB11BAF)
        self.assertGreaterEqual(self.parser.version, 24)
        self.assertGreater(len(self.parser.type_definitions), 1000)
        self.assertGreater(len(self.parser.images), 10)

        # Verify discovery of CardButton
        cb = self.parser.find_type("CardButton")
        self.assertIsNotNone(cb)
        self.assertEqual(cb.name, "CardButton")
        self.assertGreater(len(cb.fields), 5)
        self.assertEqual(cb.fields[0].name, "m_cardParent")
        self.assertEqual(cb.fields[0].offset, 0x10)

    def test_02_synthetic_object_resolution(self):
        """Verify synthetic IL2CPP object resolution from mock heap memory buffers."""
        if not self.parser:
            self.skipTest("global-metadata.dat not found on host")

        # Mock address space
        mock_memory = {}

        # 1. Allocate Mock Class Descriptor
        descriptor_addr = 0x200000000
        # Allocate class name string
        name_addr = 0x200001000
        mock_memory[name_addr] = b"CardButton\x00"

        # Il2CppClass header: name ptr at offset 16 (0x10), ns ptr at offset 24 (0x18)
        desc_buf = bytearray(48)
        struct.pack_into("<Q", desc_buf, 16, name_addr)
        struct.pack_into("<Q", desc_buf, 24, 0)
        mock_memory[descriptor_addr] = bytes(desc_buf)

        # 2. Allocate Synthetic Object Instance at 0x104b28190
        obj_addr = 0x104b28190
        obj_buf = bytearray(0x80)
        # Offset 0x0: klass* pointer
        struct.pack_into("<Q", obj_buf, 0x00, descriptor_addr)
        # Offset 0x08: monitor
        struct.pack_into("<Q", obj_buf, 0x08, 0)
        # Offset 0x10: m_cardParent -> pointer
        struct.pack_into("<Q", obj_buf, 0x10, 0x300000000)
        # Offset 0x18: m_button -> pointer
        struct.pack_into("<Q", obj_buf, 0x18, 0x300001000)
        # Offset 0x30: <ID>k__BackingField -> Int32 133700
        struct.pack_into("<i", obj_buf, 0x30, 133700)
        mock_memory[obj_addr] = bytes(obj_buf)

        # Mock reader callback
        def mock_read(addr: int, size: int) -> bytes:
            for base, chunk in mock_memory.items():
                if base <= addr < base + len(chunk):
                    offset = addr - base
                    return chunk[offset: offset + size]
            return b"\x00" * size

        resolver = RTTIResolver(self.parser)
        result = resolver.resolve_object_at_address(obj_addr, read_fn=mock_read)

        self.assertNotIn("error", result)
        self.assertEqual(result["class_name"], "CardButton")
        self.assertEqual(result["klass_descriptor"], hex(descriptor_addr))

        fields_dict = {f["name"]: f for f in result["fields"]}
        self.assertIn("m_cardParent", fields_dict)
        self.assertIn("<ID>k__BackingField", fields_dict)
        self.assertEqual(fields_dict["<ID>k__BackingField"]["value"], 133700)

        print("\n--- Synthetic Resolution Success ---")
        print(f"Address: {result['raw_address']}")
        print(f"Type: {result['resolved_type']}")
        for f in result["fields"][:6]:
            print(f"  ├─ Offset {f['offset']}: {f['name']} -> {f['value']}")

    def test_03_alignment_and_null_guards(self):
        """Verify strict 8-byte alignment and null-pointer guards."""
        if not self.parser:
            self.skipTest("global-metadata.dat not found on host")

        resolver = RTTIResolver(self.parser)

        # Misaligned address
        unaligned_res = resolver.resolve_object_at_address("0x104b28193")
        self.assertEqual(unaligned_res.get("error"), "AlignmentFault")

        # Null pointer
        def mock_read_zero(addr: int, size: int) -> bytes:
            return b"\x00" * size

        null_res = resolver.resolve_object_at_address("0x104b28190", read_fn=mock_read_zero)
        self.assertEqual(null_res.get("resolved_type"), "[NullPointer]")

    def test_04_native_mach_vm_self(self):
        """Verify native Mach VM reading memory from self process via libSystem.B.dylib."""
        pid = os.getpid()
        vm = MachVMAccess(pid)
        self.assertGreater(vm.task_port, 0)

        # Read magic number or own memory
        val = ctypes.c_uint64(0xDEADBEEFCAFEBABE)
        addr = ctypes.addressof(val)
        read_val = vm.read_uint64(addr)
        self.assertEqual(read_val, 0xDEADBEEFCAFEBABE)

        # Test region enumeration
        regions = vm.enumerate_regions(writable_only=True)
        self.assertGreater(len(regions), 0)
        vm.close()


if __name__ == "__main__":
    unittest.main()
