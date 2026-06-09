"""Utilidades para obtener información del sistema operativo y hardware."""

import platform
import subprocess


def _get_cpu_name() -> str:
    """Obtiene el nombre/modelo del CPU según el sistema operativo."""
    try:
        if platform.system() == "Darwin":
            return (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2
                )
                .decode()
                .strip()
            )
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def _get_ram_gb() -> int:
    """Obtiene la cantidad de RAM en GB según el sistema operativo."""
    try:
        if platform.system() == "Darwin":
            mem = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2)
                .decode()
                .strip()
            )
            return mem // (1024**3)
        if platform.system() == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) // (1024**2)
    except Exception:
        pass
    return 0


def build_plataforma() -> str:
    """Construye un string identificador de la plataforma: SO:CPU:RAM."""
    return f"{platform.system()}:{_get_cpu_name()}:{_get_ram_gb()}GB-RAM"
