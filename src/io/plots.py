"""Generación de figuras paper-quality (matplotlib) para el paper LaTeX."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from src.io.loader import load_all_groups

PICS_DIR = Path("src/pics")
_COLORS = [
    "#2196F3", "#F44336", "#4CAF50", "#FF9800",
    "#9C27B0", "#00BCD4", "#FFEB3B", "#795548",
]


def grid_shape(n: int) -> tuple[int, int]:
    """(rows, cols) con cols >= rows. n=6 → (2,3)."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _strats_from_df(df: pl.DataFrame) -> list[str]:
    """Columnas que son estrategias (sin sufijo _t, ni indice/alcance/mecanismo/subsys)."""
    skip = {"indice", "alcance", "mecanismo", "subsys"}
    return [c for c in df.columns if c not in skip and not c.endswith("_t")]


def _pick_reference(strategies: list[str], hint: str | None) -> str:
    if hint and hint in strategies:
        return hint
    if "phi" in strategies:
        return "phi"
    return sorted(strategies)[0]


def plot_paper_loss(
    groups_data: dict[tuple[str, str], pl.DataFrame],
    pics_dir: Path = PICS_DIR,
) -> Path:
    """Loss vs índice, una línea por estrategia, un subplot por grupo."""
    n = len(groups_data)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Comparación de pérdida por estrategia", fontsize=13, y=1.01)

    for ax_idx, ((dataset, patron), df) in enumerate(groups_data.items()):
        r, c = divmod(ax_idx, cols)
        ax = axes[r][c]
        idx = df["indice"].to_list()
        for i, strat in enumerate(sorted(_strats_from_df(df))):
            color = _COLORS[i % len(_COLORS)]
            ax.plot(idx, df[strat].to_list(), "o-", color=color, label=strat,
                    linewidth=1.5, markersize=4, alpha=0.85)
        ax.set_title(f"{dataset} | {patron}", fontsize=10)
        ax.set_xlabel("índice", fontsize=8)
        ax.set_ylabel("pérdida (Φ)", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    for ax_idx in range(n, rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    out = pics_dir / "loss_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_paper_time(
    groups_data: dict[tuple[str, str], pl.DataFrame],
    pics_dir: Path = PICS_DIR,
) -> Path:
    """Tiempo de ejecución (escala log) vs índice, un subplot por grupo."""
    n = len(groups_data)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    fig.suptitle("Comparación de tiempos de ejecución", fontsize=13, y=1.01)

    for ax_idx, ((dataset, patron), df) in enumerate(groups_data.items()):
        r, c = divmod(ax_idx, cols)
        ax = axes[r][c]
        idx = df["indice"].to_list()
        for i, strat in enumerate(sorted(_strats_from_df(df))):
            col_t = f"{strat}_t"
            if col_t not in df.columns:
                continue
            color = _COLORS[i % len(_COLORS)]
            vals = np.array(df[col_t].to_list(), dtype=float)
            vals[vals == 0] = np.nan
            ax.semilogy(idx, vals, "o-", color=color, label=strat,
                        linewidth=1.5, markersize=4, alpha=0.85)
        ax.set_title(f"{dataset} | {patron}", fontsize=10)
        ax.set_xlabel("índice", fontsize=8)
        ax.set_ylabel("tiempo (s, log)", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3, which="both")

    for ax_idx in range(n, rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    out = pics_dir / "time_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_paper_error(
    groups_data: dict[tuple[str, str], pl.DataFrame],
    reference: str | None = None,
    pics_dir: Path = PICS_DIR,
) -> Path:
    """Error relativo de cada estrategia vs referencia, un subplot por grupo."""
    n = len(groups_data)
    rows, cols = grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax_idx, ((dataset, patron), df) in enumerate(groups_data.items()):
        r, c = divmod(ax_idx, cols)
        ax = axes[r][c]
        strats = _strats_from_df(df)
        ref = _pick_reference(strats, reference)
        ref_vals = np.array(df[ref].to_list(), dtype=float)
        denom = np.where(np.abs(ref_vals) < 1e-12, 1e-12, np.abs(ref_vals))
        idx = df["indice"].to_list()
        color_i = 0
        for strat in sorted(strats):
            if strat == ref:
                continue
            rel_err = np.abs(np.array(df[strat].to_list(), dtype=float) - ref_vals) / denom * 100
            ax.plot(idx, rel_err, "o-", color=_COLORS[color_i % len(_COLORS)],
                    label=f"{strat} vs {ref}", linewidth=1.5, markersize=4, alpha=0.85)
            color_i += 1
        ax.set_title(f"{dataset} | {patron}", fontsize=10)
        ax.set_xlabel("índice", fontsize=8)
        ax.set_ylabel("error relativo (%)", fontsize=8)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Error relativo vs {reference or 'referencia (auto)'}", fontsize=13, y=1.01)

    for ax_idx in range(n, rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    out = pics_dir / "relative_error.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def generate_paper_figures(
    selected_names: list[str],
    output_dir: Path = Path("data/output"),
    pics_dir: Path = PICS_DIR,
    reference: str | None = None,
) -> list[Path]:
    """Genera las 3 figuras paper-quality y las guarda en pics_dir."""
    pics_dir.mkdir(parents=True, exist_ok=True)
    groups_data = load_all_groups(selected_names, output_dir)
    if not groups_data:
        return []
    return [
        plot_paper_loss(groups_data, pics_dir),
        plot_paper_time(groups_data, pics_dir),
        plot_paper_error(groups_data, reference, pics_dir),
    ]