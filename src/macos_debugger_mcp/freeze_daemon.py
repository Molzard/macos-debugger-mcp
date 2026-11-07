"""Background daemon thread manager for live memory address value freezing with kqueue lifecycle monitoring."""

from __future__ import annotations

import ctypes
import os
import select
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

_libc = ctypes.CDLL(None)
_mach_task_self = _libc.mach_task_self
_mach_task_self.restype = ctypes.c_uint32
_mach_port_deallocate = _libc.mach_port_deallocate
_mach_port_deallocate.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
_mach_port_deallocate.restype = ctypes.c_int

# kqueue process monitoring constants
KQ_FILTER_PROC = getattr(select, "KQ_FILTER_PROC", -5)
KQ_NOTE_EXIT = getattr(select, "KQ_NOTE_EXIT", 0x80000000)
KQ_NOTE_FORK = getattr(select, "KQ_NOTE_FORK", 0x40000000)
KQ_NOTE_EXEC = getattr(select, "KQ_NOTE_EXEC", 0x20000000)
KQ_EV_ADD = getattr(select, "KQ_EV_ADD", 1)
KQ_EV_ENABLE = getattr(select, "KQ_EV_ENABLE", 4)
KQ_EV_ONESHOT = getattr(select, "KQ_EV_ONESHOT", 16)


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
                err_str = str(e).lower()
                self.last_error = str(e)
                # If target task port is dead or process no longer exists, exit loop gracefully
                if "dead" in err_str or "invalid_task" in err_str or "kr=10" in err_str or "kr=4" in err_str:
                    break
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()


class ProcessExitWatcher(threading.Thread):
    """Monitors target process lifecycle via BSD kqueue EVFILT_PROC events."""

    def __init__(self, pid: int, debugger: Any, on_exit: Callable[[int, Any], None]):
        super().__init__(daemon=True)
        self.pid = pid
        self.debugger = debugger
        self.on_exit = on_exit
        self.stop_event = threading.Event()
        self._kq: Optional[select.kqueue] = None

    def run(self) -> None:
        try:
            self._kq = select.kqueue()
            ev = select.kevent(
                self.pid,
                filter=KQ_FILTER_PROC,
                flags=KQ_EV_ADD | KQ_EV_ENABLE | KQ_EV_ONESHOT,
                fflags=KQ_NOTE_EXIT | KQ_NOTE_FORK | KQ_NOTE_EXEC,
            )
            self._kq.control([ev], 0)
        except Exception:
            if self._kq:
                try:
                    self._kq.close()
                except Exception:
                    pass
                self._kq = None
            return

        while not self.stop_event.is_set():
            try:
                events = self._kq.control(None, 1, 0.5)
                if events:
                    for e in events:
                        if e.fflags & KQ_NOTE_EXIT:
                            self.on_exit(self.pid, self.debugger)
                            return
            except OSError:
                break

        if self._kq:
            try:
                self._kq.close()
            except Exception:
                pass
            self._kq = None

    def stop(self) -> None:
        self.stop_event.set()
        if self._kq:
            try:
                self._kq.close()
            except Exception:
                pass
            self._kq = None


_workers: Dict[str, FreezeWorker] = {}
_exit_watchers: Dict[int, ProcessExitWatcher] = {}
_lock = threading.Lock()


def _handle_process_exit(pid: int, debugger: Any) -> None:
    """Invoked when kqueue detects NOTE_EXIT on a target process."""
    with _lock:
        # 1. Gracefully stop and clean up all freeze workers associated with this PID
        stopped_ids = [fid for fid, w in _workers.items() if w.pid == pid]
        for fid in stopped_ids:
            _workers[fid].stop()
            del _workers[fid]

        # 2. Stop and remove exit watcher
        watcher = _exit_watchers.pop(pid, None)
        if watcher:
            watcher.stop()

        # 3. Deallocate Mach task port in kernel namespace to prevent port-right leaks
        if hasattr(debugger, "cleanup_task_port"):
            debugger.cleanup_task_port(pid)
        else:
            task_port = getattr(debugger, "task_ports", {}).pop(pid, None)
            if task_port and task_port != _mach_task_self():
                try:
                    _mach_port_deallocate(_mach_task_self(), task_port)
                except Exception:
                    pass


def freeze_address_value(
    debugger: Any,
    pid: int,
    address: str,
    bytes_hex: str,
    interval_ms: int = 100,
) -> Dict[str, Any]:
    """Lock memory address to a fixed value using a non-blocking background daemon thread.
    
    Includes kqueue process lifecycle monitoring to prevent Mach task port dead-name leaks.
    """
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

        # Register kqueue exit watcher for this PID if not already active
        if pid not in _exit_watchers and pid != os.getpid():
            watcher = ProcessExitWatcher(pid, debugger, _handle_process_exit)
            _exit_watchers[pid] = watcher
            watcher.start()

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
            pids = set(w.pid for w in _workers.values())
            for w in _workers.values():
                w.stop()
            _workers.clear()

            # Stop watchers for all PIDs
            for pid in pids:
                watcher = _exit_watchers.pop(pid, None)
                if watcher:
                    watcher.stop()

            return {
                "status": "unfrozen",
                "terminated_count": len(stopped_ids),
                "freeze_ids": stopped_ids,
            }

        if freeze_id not in _workers:
            raise KeyError(f"No active freeze worker found with ID '{freeze_id}'")

        worker = _workers.pop(freeze_id)
        worker.stop()

        # If no more workers for this pid, stop the watcher
        pid = worker.pid
        has_remaining = any(w.pid == pid for w in _workers.values())
        if not has_remaining:
            watcher = _exit_watchers.pop(pid, None)
            if watcher:
                watcher.stop()

    return {
        "status": "unfrozen",
        "freeze_id": freeze_id,
        "total_writes": worker.write_count,
    }
