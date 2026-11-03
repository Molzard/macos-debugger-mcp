"""Background daemon thread manager for live memory address value freezing."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional


class FreezeWorker(threading.Thread):
    def __init__(
        self,
        debugger: Any,
        pid: int,
        address: str,
        bytes_hex: str,
        interval_ms: int = 100,
    ):
        super().__init__(daemon=True)
        self.debugger = debugger
        self.pid = pid
        self.address = address
        self.bytes_hex = bytes_hex
        self.interval = max(interval_ms, 10) / 1000.0
        self.stop_event = threading.Event()
        self.last_error: Optional[str] = None
        self.write_count = 0

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                self.debugger.write_memory(self.pid, self.address, self.bytes_hex)
                self.write_count += 1
            except Exception as e:
                self.last_error = str(e)
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()


_workers: Dict[str, FreezeWorker] = {}
_lock = threading.Lock()


def freeze_address_value(
    debugger: Any,
    pid: int,
    address: str,
    bytes_hex: str,
    interval_ms: int = 100,
) -> Dict[str, Any]:
    """Lock memory address to a fixed value using a non-blocking background daemon thread."""
    # Validate address and payload upfront
    clean_hex = bytes_hex.strip().replace(" ", "").replace("0x", "")
    bytes.fromhex(clean_hex)

    # Apply initial write immediately to verify validity
    debugger.write_memory(pid, address, clean_hex)

    freeze_id = f"frz_{uuid.uuid4().hex[:8]}"
    worker = FreezeWorker(debugger, pid, address, clean_hex, interval_ms)

    with _lock:
        _workers[freeze_id] = worker
        worker.start()

    return {
        "status": "frozen",
        "freeze_id": freeze_id,
        "pid": pid,
        "address": address,
        "bytes_hex": clean_hex,
        "interval_ms": interval_ms,
    }


def unfreeze_address_value(freeze_id: str) -> Dict[str, Any]:
    """Release and terminate active background memory freeze worker(s)."""
    with _lock:
        if freeze_id.lower() == "all":
            stopped_ids = list(_workers.keys())
            for w in _workers.values():
                w.stop()
            _workers.clear()
            return {
                "status": "unfrozen",
                "terminated_count": len(stopped_ids),
                "freeze_ids": stopped_ids,
            }

        if freeze_id not in _workers:
            raise KeyError(f"No active freeze worker found with ID '{freeze_id}'")

        worker = _workers.pop(freeze_id)
        worker.stop()

    return {
        "status": "unfrozen",
        "freeze_id": freeze_id,
        "total_writes": worker.write_count,
    }
