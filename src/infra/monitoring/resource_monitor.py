"""Resource monitoring with zero-overhead aggregate stats (CPU, memory, GPU).

Uses strategy pattern: detects platform and enables appropriate monitoring.
All stats are collected from kernel/OS APIs - NO active sampling overhead.
"""

from __future__ import annotations

import resource
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class ResourceStats:
    """Aggregate resource usage for a single execution."""

    tiempo_wall_s: float = 0.0
    tiempo_cpu_s: float = 0.0
    cpu_user_s: float = 0.0
    cpu_sys_s: float = 0.0
    mem_rss_mb: float = 0.0
    gpu_mem_mb: float = 0.0


class ResourceMonitorStrategy(ABC):
    """Abstract strategy for resource monitoring."""

    @abstractmethod
    def before_exec(self) -> None:
        """Capture baseline state before execution."""
        pass

    @abstractmethod
    def after_exec(self) -> ResourceStats:
        """Capture end state and compute deltas."""
        pass


class BasicMonitorStrategy(ResourceMonitorStrategy):
    """CPU/memory monitoring using standard library (resource module)."""

    _start_rusage: Optional[resource.struct_rusage] = None

    def before_exec(self) -> None:
        self._start_rusage = resource.getrusage(resource.RUSAGE_SELF)

    def after_exec(self) -> ResourceStats:
        end_rusage = resource.getrusage(resource.RUSAGE_SELF)
        stats = ResourceStats()

        stats.cpu_user_s = round(end_rusage.ru_utime - self._start_rusage.ru_utime, 6)
        stats.cpu_sys_s = round(end_rusage.ru_stime - self._start_rusage.ru_stime, 6)
        stats.tiempo_cpu_s = round(stats.cpu_user_s + stats.cpu_sys_s, 6)

        import platform
        if platform.system() == "Darwin":
            start_kb = self._start_rusage.ru_maxrss / 1024
            end_kb = end_rusage.ru_maxrss / 1024
        else:
            start_kb = self._start_rusage.ru_maxrss
            end_kb = end_rusage.ru_maxrss
        stats.mem_rss_mb = round((end_kb - start_kb) / 1024, 3)

        return stats


class GPUMonitorStrategy(ResourceMonitorStrategy):
    """GPU monitoring using pynvml (NVIDIA only). Falls back to basic if unavailable."""

    _basic: BasicMonitorStrategy
    _gpu_handle: Optional[object] = None
    _start_gpu_mem_mb: int = 0

    def __init__(self) -> None:
        self._basic = BasicMonitorStrategy()
        self._init_gpu()

    def _init_gpu(self) -> None:
        if not NVML_AVAILABLE:
            return
        try:
            pynvml.nvmlInit()
            self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._gpu_handle = None

    def before_exec(self) -> None:
        self._basic.before_exec()
        if self._gpu_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                self._start_gpu_mem_mb = mem_info.used // (1024 * 1024)
            except Exception:
                self._start_gpu_mem_mb = 0

    def after_exec(self) -> ResourceStats:
        stats = self._basic.after_exec()

        if self._gpu_handle is not None:
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                end_gpu_mem_mb = mem_info.used // (1024 * 1024)
                stats.gpu_mem_mb = max(0, end_gpu_mem_mb - self._start_gpu_mem_mb)
            except Exception:
                stats.gpu_mem_mb = 0.0

        return stats


class ResourceMonitor:
    """Context class that manages resource monitoring for a single execution.

    Usage:
        monitor = ResourceMonitor()
        monitor.start()
        # ... run algorithm ...
        stats = monitor.stop()
    """

    _strategy: ResourceMonitorStrategy

    def __init__(self) -> None:
        self._strategy = GPUMonitorStrategy() if NVML_AVAILABLE else BasicMonitorStrategy()
        self._wall_start: float = 0.0

    def start(self) -> None:
        """Begin monitoring."""
        self._wall_start = time.perf_counter()
        self._strategy.before_exec()

    def stop(self) -> ResourceStats:
        """Stop monitoring and return stats."""
        wall_end = time.perf_counter()
        stats = self._strategy.after_exec()
        stats.tiempo_wall_s = round(wall_end - self._wall_start, 6)
        return stats