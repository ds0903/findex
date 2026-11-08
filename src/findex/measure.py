import gc
import io
import sys
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Measured[T]:
    result: T
    elapsed: float  # секунди
    peak_bytes: int | None  # None, якщо tracemalloc вимкнено


def measure[T](func: Callable[[], T], trace_memory: bool = True) -> Measured[T]:
    """Запускає func, міряє час і піковий обсяг пам'яті (tracemalloc).

    З увімкненим tracemalloc код працює в рази повільніше, тож час з ним
    порівнювати з іншими запусками не можна.
    """
    gc.collect()
    if trace_memory:
        tracemalloc.start()
    start = time.perf_counter()
    try:
        result = func()
        elapsed = time.perf_counter() - start
        peak = tracemalloc.get_traced_memory()[1] if trace_memory else None
    finally:
        if trace_memory:
            tracemalloc.stop()
    return Measured(result=result, elapsed=elapsed, peak_bytes=peak)


def mb(n_bytes: float) -> float:
    return n_bytes / 1024 / 1024


def peak_rss_mb() -> float:
    """Пік оперативної пам'яті поточного процесу, МБ (на Windows це пік робочого набору)."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
        return mb(counters.PeakWorkingSetSize)
    import resource

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # КБ на Linux, байти на macOS
    return mb(rss * (1 if sys.platform == "darwin" else 1024))


def setup_console() -> None:
    # у windows консоль часто cp1251, а в корпусі є будь-які символи
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
