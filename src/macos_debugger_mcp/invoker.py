"""In-process native method invocation engine using Frida Core with fallback evaluation."""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional, Union

from .debugger import MacOSDebugger
from .utils import parse_address


def invoke_native_method(
    debugger: MacOSDebugger,
    pid: int,
    function_address: str,
    args: Optional[List[Dict[str, Any]]] = None,
    return_type: str = "void",
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Invoke a native C/C++/IL2CPP function at a given address in target process using dynamic runtime evaluation.

    Args:
        debugger: MacOSDebugger instance for state & inspection.
        pid: Target process ID.
        function_address: Hex memory address of function entry point (e.g. '0x102b48000').
        args: List of typed argument objects: [{"type": "int32"|"int64"|"float"|"double"|"pointer"|"string", "value": ...}].
        return_type: Expected return type ('void', 'int32', 'int64', 'float', 'double', 'pointer', 'string').
        timeout: Execution timeout in seconds (default: 5.0).

    Returns:
        Dict with execution status, engine used, function address, return type, and resulting value.
    """
    if args is None:
        args = []

    func_addr_int = parse_address(function_address)
    hex_addr = f"0x{func_addr_int:x}"

    # 1. Primary Engine: Frida Dynamic Instrumentation Core
    try:
        import frida  # type: ignore

        session = frida.attach(pid)
        js_code = """
        rpc.exports = {
            invoke: function(funcAddr, retType, args) {
                // Safety Guard: Verify memory region executability
                const p = ptr(funcAddr);
                const r = Process.findRangeByAddress(p);
                if (!r || r.protection.indexOf("x") === -1) {
                    throw new Error("Target address " + funcAddr + " is not in an executable memory segment (protection: " + (r ? r.protection : "unmapped") + ")");
                }

                const typeMap = {
                    "void": "void",
                    "int": "int",
                    "int32": "int32",
                    "uint32": "uint32",
                    "int64": "int64",
                    "uint64": "uint64",
                    "float": "float",
                    "double": "double",
                    "pointer": "pointer",
                    "bool": "bool",
                    "string": "pointer"
                };

                const fridaRetType = typeMap[retType.toLowerCase()] || "pointer";
                const fridaArgTypes = [];
                const parsedArgs = [];

                for (let i = 0; i < args.length; i++) {
                    const arg = args[i];
                    const aType = (arg.type || "pointer").toLowerCase();
                    const aVal = arg.value;
                    fridaArgTypes.push(typeMap[aType] || "pointer");

                    if (aType === "string") {
                        parsedArgs.push(Memory.allocUtf8String(aVal));
                    } else if (aType === "string16" || aType === "utf16") {
                        parsedArgs.push(Memory.allocUtf16String(aVal));
                    } else if (aType === "pointer") {
                        parsedArgs.push(typeof aVal === "string" ? ptr(aVal) : ptr(aVal.toString()));
                    } else if (aType === "int64" || aType === "uint64") {
                        parsedArgs.push(typeof aVal === "string" ? int64(aVal) : int64(aVal.toString()));
                    } else {
                        parsedArgs.push(aVal);
                    }
                }

                const fn = new NativeFunction(p, fridaRetType, fridaArgTypes);
                const rawRes = fn(...parsedArgs);

                if (retType.toLowerCase() === "void") {
                    return null;
                } else if (retType.toLowerCase() === "string") {
                    return rawRes.isNull() ? null : rawRes.readUtf8String();
                } else if (retType.toLowerCase() === "pointer") {
                    return rawRes ? rawRes.toString() : "0x0";
                } else {
                    return rawRes;
                }
            }
        };
        """
        script = session.create_script(js_code)
        script.load()

        start_time = time.time()
        # Execute RPC call
        result = script.exports_sync.invoke(hex_addr, return_type, args)
        duration_ms = (time.time() - start_time) * 1000.0

        session.detach()

        return {
            "status": "success",
            "pid": pid,
            "function_address": hex_addr,
            "engine": "frida",
            "return_type": return_type,
            "duration_ms": round(duration_ms, 2),
            "result": result,
        }

    except Exception as frida_err:
        err_msg = str(frida_err)
        # If it's a safety violation (non-executable), propagate directly
        if "not in an executable memory segment" in err_msg:
            raise PermissionError(err_msg)

        # 2. Fallback Engine: LLDB Expression Evaluation
        try:
            lldb_res = _fallback_lldb_invoke(pid, hex_addr, args, return_type, timeout)
            return {
                "status": "success",
                "pid": pid,
                "function_address": hex_addr,
                "engine": "lldb",
                "return_type": return_type,
                "result": lldb_res,
            }
        except Exception as lldb_err:
            raise RuntimeError(
                f"Invocation failed across both engines. Frida error: {frida_err}. LLDB error: {lldb_err}"
            )


def _fallback_lldb_invoke(
    pid: int,
    func_hex: str,
    args: List[Dict[str, Any]],
    return_type: str,
    timeout: float,
) -> Any:
    """Secondary fallback invocation using LLDB expression evaluation."""
    c_args = []
    for a in args:
        atype = a.get("type", "pointer")
        aval = a.get("value")
        if atype == "string":
            c_args.append(f'"{aval}"')
        elif atype in ["pointer", "int64", "uint64"]:
            c_args.append(f"(void*){aval}")
        else:
            c_args.append(str(aval))

    args_str = ", ".join(c_args)
    ret_cast = "void*" if return_type in ["pointer", "string"] else return_type
    expr = f"(( {ret_cast} (*)({', '.join(['...'])}) ){func_hex})({args_str})"

    import subprocess
    cmd = [
        "lldb",
        "--batch",
        "-p", str(pid),
        "-o", f"expr {expr}",
        "-o", "detach",
        "-o", "quit"
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"LLDB expression failed: {p.stderr}")
    return p.stdout.strip()
