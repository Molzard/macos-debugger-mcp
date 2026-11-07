"""Non-destructive live method interception telemetry (hook_and_log_method) via Frida Dynamic Instrumentation."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .utils import parse_address, strip_pac


def hook_and_log_method(
    debugger: Any,
    pid: int,
    function_address: str,
    capture_count: int = 10,
    timeout_seconds: float = 10.0,
    include_backtrace: bool = True,
    mock_invocations: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Attach temporary non-destructive live hook via Frida to record arguments (x0-x7), floats (d0-d7), return value, and backtrace."""
    addr = strip_pac(parse_address(function_address))
    hex_addr = f"0x{addr:x}"

    clamped_count = max(1, min(capture_count, 50))
    clamped_timeout = max(0.5, min(timeout_seconds, 30.0))

    if mock_invocations is not None:
        return {
            "status": "success",
            "pid": pid,
            "function_address": hex_addr,
            "captured_invocations": len(mock_invocations),
            "invocations": mock_invocations,
        }

    try:
        import frida  # type: ignore
    except ImportError:
        raise RuntimeError("frida library is not installed in the environment")

    invocations: List[Dict[str, Any]] = []
    error_msg: Optional[str] = None
    is_ready = False
    is_done = False

    def on_message(message: Dict[str, Any], data: Any) -> None:
        nonlocal is_ready, is_done, error_msg
        if message.get("type") == "send":
            payload = message.get("payload", {})
            msg_type = payload.get("type")
            if msg_type == "ready":
                is_ready = True
            elif msg_type == "invocation":
                invocations.append(payload.get("payload", {}))
                if len(invocations) >= clamped_count:
                    is_done = True
            elif msg_type == "complete":
                is_done = True
            elif msg_type == "error":
                error_msg = payload.get("message")
        elif message.get("type") == "error":
            error_msg = message.get("description", str(message))

    js_template = f"""
    const targetPtr = ptr("{hex_addr}");
    let captured = 0;
    const maxCaptures = {clamped_count};
    const includeBt = {"true" if include_backtrace else "false"};

    try {{
        // Auto-attach thread to IL2CPP runtime if available
        const pDomainGet = Module.findGlobalExportByName("il2cpp_domain_get");
        const pThreadAttach = Module.findGlobalExportByName("il2cpp_thread_attach");
        if (pDomainGet && pThreadAttach) {{
            const domainGet = new NativeFunction(pDomainGet, "pointer", []);
            const threadAttach = new NativeFunction(pThreadAttach, "pointer", ["pointer"]);
            threadAttach(domainGet());
        }}
    }} catch(e) {{}}

    try {{
        Interceptor.attach(targetPtr, {{
            onEnter: function(args) {{
                this.invId = ++captured;
                this.startTime = Date.now();
                this.args = {{}};
                for (let i = 0; i < 8; i++) {{
                    try {{
                        this.args["x" + i] = args[i].toString();
                    }} catch(e) {{
                        this.args["x" + i] = "0x0";
                    }}
                }}
                this.floatArgs = {{}};
                for (let i = 0; i < 8; i++) {{
                    try {{
                        const regName = "d" + i;
                        if (this.context[regName] !== undefined) {{
                            this.floatArgs[regName] = this.context[regName];
                        }}
                    }} catch(e) {{}}
                }}
                this.backtrace = [];
                if (includeBt) {{
                    try {{
                        this.backtrace = Thread.backtrace(this.context, Backtracer.ACCURATE)
                            .map(DebugSymbol.fromAddress)
                            .map(function(s) {{ return s.toString(); }});
                    }} catch(e) {{}}
                }}
            }},
            onLeave: function(retval) {{
                const duration = Date.now() - this.startTime;
                send({{
                    type: "invocation",
                    payload: {{
                        invocation_id: this.invId,
                        timestamp_ms: this.startTime,
                        execution_duration_ms: duration,
                        args: this.args,
                        float_args: this.floatArgs,
                        return_value: retval ? retval.toString() : "0x0",
                        backtrace: this.backtrace
                    }}
                }});
                if (captured >= maxCaptures) {{
                    send({{type: "complete"}});
                }}
            }}
        }});
        send({{type: "ready"}});
    }} catch(e) {{
        send({{type: "error", message: e.toString()}});
    }}
    """

    session = None
    try:
        session = frida.attach(pid)
        script = session.create_script(js_template)
        script.on("message", on_message)
        script.load()

        start_time = time.time()
        while not is_done and (time.time() - start_time) < clamped_timeout:
            if error_msg:
                break
            time.sleep(0.05)

    except Exception as ex:
        raise RuntimeError(f"Frida interception failed on PID {pid}: {ex}")
    finally:
        if session:
            try:
                session.detach()
            except Exception:
                pass

    if error_msg:
        raise RuntimeError(f"Interceptor failed at {hex_addr}: {error_msg}")

    return {
        "status": "success",
        "pid": pid,
        "function_address": hex_addr,
        "captured_invocations": len(invocations),
        "invocations": invocations,
    }
