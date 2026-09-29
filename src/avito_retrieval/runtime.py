from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import dataclass

import psutil
import torch


@dataclass(frozen=True)
class RuntimeInfo:
    device: str
    cpu_threads: int
    ram_gb: float
    platform: str
    mps_name: str | None


def detect_cpu_threads() -> int:
    try:
        raw = subprocess.check_output(["sysctl", "-n", "hw.logicalcpu"], text=True).strip()
        return int(raw)
    except Exception:
        return int(os.cpu_count() or 4)


def choose_device(requested: str = "auto") -> str:
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in the installed PyTorch")
        return "mps"
    if requested == "cpu":
        return "cpu"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def configure_runtime(requested_device: str = "auto", cpu_threads: str | int = "auto") -> RuntimeInfo:
    detected = detect_cpu_threads()
    threads = detected if cpu_threads == "auto" else int(cpu_threads)
    threads = max(1, min(threads, detected))
    worker_threads = max(1, int(round(threads * 0.8)))

    # These values are set before expensive NumPy/LightGBM work starts in a fresh CLI process.
    os.environ["AVITO_CPU_THREADS"] = str(worker_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(worker_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(worker_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(worker_threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(worker_threads))

    torch.set_num_threads(worker_threads)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = choose_device(requested_device)
    mps_name = None
    if device == "mps":
        try:
            mps_name = torch.backends.mps.get_name()
        except Exception:
            mps_name = "Apple Metal GPU"
    return RuntimeInfo(device, worker_threads, psutil.virtual_memory().total / 1024**3, platform.platform(), mps_name)


def print_runtime(info: RuntimeInfo) -> None:
    print(f"platform: {info.platform}")
    print(f"RAM: {info.ram_gb:.1f} GB")
    print(f"CPU threads: {info.cpu_threads}")
    print(f"torch device: {info.device}")
    if info.mps_name:
        print(f"MPS device: {info.mps_name}")
