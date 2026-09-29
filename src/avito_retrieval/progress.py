from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass

from tqdm.auto import tqdm


@dataclass
class Timer:
    name: str
    start: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        print(f"\n[{self.name}] started", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = time.perf_counter() - self.start
        if exc is None:
            print(f"[{self.name}] done in {elapsed:.1f}s", flush=True)
        else:
            print(f"[{self.name}] failed after {elapsed:.1f}s", flush=True)
        return False


def progress(iterable, *, total=None, desc="", unit="it"):
    return tqdm(iterable, total=total, desc=desc, unit=unit, dynamic_ncols=True, mininterval=1.5)


@contextlib.contextmanager
def heartbeat(name: str, interval: float = 15.0):
    stop = threading.Event()
    started = time.perf_counter()

    def worker():
        while not stop.wait(interval):
            elapsed = time.perf_counter() - started
            print(f"[{name}] still running; elapsed={elapsed:.0f}s", flush=True)

    thread = threading.Thread(target=worker, name=f"heartbeat:{name}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.5)
