"""Automated ARM64 Array of Bytes (AoB) pattern signature generator with Capstone instruction-aware masking."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Set

try:
    from capstone import CS_ARCH_ARM64, CS_MODE_ARM, Cs
except ImportError:
    Cs = None  # type: ignore
    CS_ARCH_ARM64 = None  # type: ignore
    CS_MODE_ARM = None  # type: ignore

from .utils import parse_address


def mask_arm64_instruction(insn: Any, adrp_regs: Set[str]) -> List[str]:
    """Mask relocation and PC-relative bytes to wildcards ('??') while preserving opcodes and non-volatile registers."""
    mnemonic = insn.mnemonic.lower()
    raw = insn.bytes
    if len(raw) != 4:
        return [f"{b:02X}" for b in raw]

    b0, b1, b2, b3 = raw[0], raw[1], raw[2], raw[3]

    # 1. ADRP / ADR (Page-relative addressing)
    if mnemonic in ["adrp", "adr"]:
        ops = insn.op_str.split(",")
        if ops:
            dst = ops[0].strip().lower()
            adrp_regs.add(dst)
        return ["??", "??", "??", "??"]

    # 2. Unconditional Branch B & Branch-with-link BL
    if mnemonic == "b":
        return ["??", "??", "??", "14"]
    if mnemonic == "bl":
        return ["??", "??", "??", "94"]

    # 3. Conditional branches B.<cond> (opcode byte 0x54, condition in b0)
    if mnemonic.startswith("b."):
        return [f"{b0:02X}", "??", "??", "54"]

    # 4. Compare and Branch (CBZ / CBNZ)
    if mnemonic in ["cbz", "cbnz"]:
        return ["??", "??", "??", f"{b3:02X}"]

    # 5. Test and Branch (TBZ / TBNZ)
    if mnemonic in ["tbz", "tbnz"]:
        return ["??", "??", "??", f"{b3:02X}"]

    # 6. LDR / STR / ADD / SUB with immediate offset from register loaded by ADRP
    # Byte 0 contains target/source register and Rn bits, Byte 3 contains opcode.
    # Immediate relocation bits reside in Bytes 1 and 2.
    if mnemonic in ["ldr", "str", "ldrb", "strb", "ldrh", "strh", "add", "sub"]:
        uses_adrp = any(reg in insn.op_str.lower() for reg in adrp_regs)
        if uses_adrp:
            ops = [op.strip().lower() for op in insn.op_str.split(",")]
            if ops and mnemonic in ["add", "sub"]:
                dst = ops[0].strip().lower()
                adrp_regs.add(dst)
            return [f"{b0:02X}", "??", "??", f"{b3:02X}"]

    # Normal non-relocation instruction: preserve all bytes
    return [f"{b0:02X}", f"{b1:02X}", f"{b2:02X}", f"{b3:02X}"]


def tokens_to_regex(tokens: List[str]) -> re.Pattern[bytes]:
    """Compile AoB pattern tokens into a byte regular expression."""
    pattern_bytes = []
    for tok in tokens:
        if tok == "??":
            pattern_bytes.append(b".")
        else:
            pattern_bytes.append(bytes([int(tok, 16)]))
    return re.compile(b"".join(pattern_bytes), re.DOTALL)


def generate_arm64_aob_signature(
    debugger: Any,
    pid: int,
    function_address: str,
    scan_length: int = 64,
    text_segment_bytes: Optional[bytes] = None,
    text_segment_base: Optional[int] = None,
) -> dict[str, Any]:
    """Generate update-resilient Array of Bytes (AoB) pattern signature from ARM64 machine code with PC-relative relocation wildcarding."""
    if Cs is None:
        raise RuntimeError("capstone disassembly library is not installed in current environment")

    addr = parse_address(function_address)
    scan_len = max(scan_length, 4)

    # 1. Read instruction bytes at function_address
    raw_code = b""
    if text_segment_bytes is not None and text_segment_base is not None:
        if text_segment_base <= addr < text_segment_base + len(text_segment_bytes):
            offset = addr - text_segment_base
            raw_code = text_segment_bytes[offset : offset + scan_len]
    else:
        if hasattr(debugger, "read_raw_bytes"):
            raw_code = debugger.read_raw_bytes(pid, addr, scan_len)
        else:
            res = debugger.read_memory(pid, hex(addr), scan_len, fmt="hex")
            hex_data = res.get("hex", "")
            raw_code = bytes.fromhex(hex_data) if hex_data else b""

    if not raw_code:
        raise ValueError(f"Could not read executable code at address 0x{addr:x}")

    # 2. Disassemble via Capstone ARM64
    md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
    md.detail = True
    instructions = list(md.disasm(raw_code, addr))
    if not instructions:
        raise ValueError(f"No valid ARM64 instructions could be disassembled at 0x{addr:x}")

    # 3. Mask instructions per 4-byte instruction boundary
    adrp_regs: Set[str] = set()
    instruction_token_groups: List[List[str]] = []
    for insn in instructions:
        tokens = mask_arm64_instruction(insn, adrp_regs)
        instruction_token_groups.append(tokens)

    # 4. Locate __TEXT / executable segment for uniqueness scanning
    exec_bytes = text_segment_bytes
    exec_base = text_segment_base

    if exec_bytes is None:
        regions_info = debugger.inspect_memory_regions(pid, limit=1000)
        regions = regions_info.get("regions", [])
        target_region = None

        # Find region enclosing target address
        for r in regions:
            r_start = int(r.get("start_address") or r.get("start", "0"), 16)
            r_size = r.get("size_bytes") or r.get("size", 0)
            if r_start <= addr < r_start + r_size:
                prot = r.get("protection") or r.get("protection_str", "")
                if "x" in prot:
                    target_region = (r_start, r_size)
                    break

        if target_region:
            exec_base, exec_size = target_region
            if hasattr(debugger, "read_raw_bytes"):
                exec_bytes = debugger.read_raw_bytes(pid, exec_base, exec_size)
            else:
                res = debugger.read_memory(pid, hex(exec_base), exec_size, fmt="hex")
                hex_data = res.get("hex", "")
                exec_bytes = bytes.fromhex(hex_data) if hex_data else None

    # 5. Expand pattern by 4 bytes (per instruction) until exactly 1 unique match is found
    if exec_bytes and exec_base is not None:
        for insn_count in range(1, len(instruction_token_groups) + 1):
            selected_tokens: List[str] = []
            for grp in instruction_token_groups[:insn_count]:
                selected_tokens.extend(grp)

            pat = tokens_to_regex(selected_tokens)
            matches = [m.start() for m in pat.finditer(exec_bytes)]

            if len(matches) == 1:
                matched_addr = exec_base + matches[0]
                return {
                    "status": "success",
                    "signature": " ".join(selected_tokens),
                    "length": len(selected_tokens),
                    "unique_match": True,
                    "matched_address": f"0x{matched_addr:x}",
                }

    # Fallback if segment not scanned or match remained non-unique
    fallback_tokens: List[str] = []
    for grp in instruction_token_groups:
        fallback_tokens.extend(grp)

    return {
        "status": "success",
        "signature": " ".join(fallback_tokens),
        "length": len(fallback_tokens),
        "unique_match": (exec_bytes is not None and len(instruction_token_groups) > 0),
        "matched_address": f"0x{addr:x}",
    }
