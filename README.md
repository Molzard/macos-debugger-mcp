# macos-debugger-mcp

Model Context Protocol (MCP) server for LLDB-based process debugging, Mach VM memory telemetry, and dynamic instrumentation on macOS (Apple Silicon ARM64).

## Architecture

The server exposes low-level macOS debugging capabilities to MCP clients:
- **Process Telemetry:** Native `libproc` process discovery with architecture detection (native ARM64 / Rosetta x86_64) and bundle ID resolution.
- **Mach Kernel Bridge:** Direct `ctypes` bindings for `task_for_pid`, `mach_vm_region_recurse`, `mach_vm_read_overwrite`, `mach_vm_write`, and `mach_vm_protect`.
- **Apple LLDB Engine:** `lldb.SBDebugger` target attachment and hardware watchpoint instrumentation (`SBTarget.WatchAddress`) with resilient headless scripting fallback via `/usr/bin/python3`.
- **Security Diagnostic:** Automatic diagnosis for task port permission failures (`KERN_FAILURE`, SIP, and `get-task-allow`).

## Tools Implemented (15 MCP Tools)

### Core Debugging & Memory Inspection
1. `list_target_processes`: Enumerate running processes with PID, executable path, architecture, and bundle ID.
2. `attach_debugger`: Attach LLDB debugger and acquire Mach task port for target PID.
3. `inspect_memory_regions`: Retrieve virtual memory map, segment sizes, user tags (Heap, Stack, Dyld), and protection flags (`r-x`, `rw-`).
4. `search_memory`: Scan memory regions (`rw-` heap or all readable) for values (`int32`, `int64`, `float`, `double`, `bytes`, `string`) returning candidate addresses.
5. `read_memory_bytes`: Read memory and unpack as `hex`, `int32`, `int64`, `float`, `double`, or `string`.
6. `write_memory_bytes`: Patch memory addresses with hex payloads (with automatic page protection override if read-only).
7. `set_watchpoint`: Set ARM64 hardware watchpoints (`read`, `write`, `read_write`) with resilient LLDB bridge.

### Unity IL2CPP Semantic Profiling
8. `resolve_object_at_address`: Parse IL2CPP runtime `Il2CppClass*` at address and reconstruct full object graph (fields, types, child elements).
9. `find_active_instances_of_class`: Search writable heap for all active allocated instances of a given C# class.

### Remote Native Execution
10. `invoke_native_method`: Call a native C/C++/IL2CPP function at runtime using Frida Core with LLDB fallback.

### Heap Snapshots, State Freezing, & Dylib Exporter
11. `take_memory_snapshot`: Capture a point-in-time snapshot of process writable heap regions.
12. `diff_memory_snapshots`: Compare two heap snapshots to find mutated values (`changed`, `increased`, `decreased`, `exact_difference`).
13. `freeze_address_value`: Lock a memory address to a fixed value with a background assertion worker thread.
14. `unfreeze_address_value`: Release an active memory freeze worker thread.
15. `export_dylib_patch`: Compile standalone ARM64 C/Objective-C patch into a distributable `.dylib` via `clang`.

## Requirements & Permissions

- macOS on Apple Silicon (ARM64)
- Python 3.11+
- For debugging external third-party or system processes, macOS requires:
  - Developer Mode enabled: `sudo DevToolsSecurity -enable`
  - Or running the server with `sudo` / task port entitlements.
  - Target applications compiled with `get-task-allow` entitlement (standard debug builds) can be debugged without root.

## Installation

Using `uv`:

```bash
git clone https://github.com/Molzard/macos-debugger-mcp.git
cd macos-debugger-mcp
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## MCP Configuration

Add to your MCP client config (e.g., Claude Desktop, Hermes, Cursor):

```json
{
  "mcpServers": {
    "macos-debugger": {
      "command": "/PATH/TO/macos-debugger-mcp/.venv/bin/python",
      "args": ["-m", "macos_debugger_mcp.server"]
    }
  }
}
```

Or run directly with `uv`:

```json
{
  "mcpServers": {
    "macos-debugger": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/PATH/TO/macos-debugger-mcp",
        "macos-debugger-mcp"
      ]
    }
  }
}
```

## Running Tests

Run the test suite to verify Mach VM memory scanning and debugger bridge:

```bash
python tests/test_debugger.py
```
