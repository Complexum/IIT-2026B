"""Helpers compartidos para la generación de gráficos plotly."""

import math
import subprocess
from pathlib import Path

import polars as pl

from src.tui.analysis.compare import RESULTADOS_DIR

__all__ = [
    "RESULTADOS_DIR",
    "_COLORS",
    "_grid_shape",
    "_strats",
    "_pick_ref",
    "_subplot_titles",
    "_hover_text",
    "open_plot",
    "RESOURCE_METRICS",
]


_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FFEB3B", "#795548",
]


RESOURCE_METRICS = [
    ("cpu_user_s", "CPU user (s)"),
    ("cpu_sys_s", "CPU system (s)"),
    ("mem_rss_mb", "Memory RSS (MB)"),
    ("gpu_mem_mb", "GPU memory (MB)"),
]


def _hover_text(idx, a, b, diff, rel, label_a, label_b):
    return [
        f"idx={i}<br>{label_a}: {va:.6f}<br>{label_b}: {vb:.6f}"
        f"<br>diff={d:.6f}<br>rel={r:.2f}%"
        for i, va, vb, d, r in zip(idx, a, b, diff, rel)
    ]


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _strats(df: pl.DataFrame) -> list[str]:
    skip = {"indice", "alcance", "mecanismo", "subsys"}
    resource_suffixes = {"_cpu_user_s", "_cpu_sys_s", "_mem_rss_mb", "_gpu_mem_mb"}
    return [
        c for c in df.columns
        if c not in skip
        and not c.endswith("_t")
        and not any(c.endswith(suf) for suf in resource_suffixes)
    ]


def _pick_ref(strategies: list[str], hint: str | None) -> str:
    if hint and hint in strategies:
        return hint
    return "phi" if "phi" in strategies else sorted(strategies)[0]


def _subplot_titles(groups_data: dict) -> list[str]:
    return [f"{ds} | {pat}" for (ds, pat) in groups_data]


def open_plot(path: Path) -> None:
    """Abre el HTML generado en el navegador predeterminado."""
    subprocess.Popen(["open", str(path)])
