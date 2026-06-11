"""Lógica de comparación de resultados CSV usando polars."""

from itertools import combinations
from pathlib import Path

import polars as pl
from rich import box as rich_box
from rich.table import Table

from src.io.manager import listar_resultados
from src.tui.run.helpers import cargar_programa, extract_program_name

RESULTADOS_DIR = Path("data/output")


def load_result(name: str) -> pl.DataFrame:
    """Carga un CSV de resultado y renombra las columnas de interés."""
    return (
        pl.read_csv(RESULTADOS_DIR / f"{name}.csv", infer_schema_length=0)
        .select(["indice", "perdida", "tiempo_wall_s"])
        .rename({"perdida": f"perdida_{name}", "tiempo_wall_s": f"tiempo_{name}"})
        .with_columns(
            pl.col("indice").cast(pl.Int64),
            pl.col(f"perdida_{name}").cast(pl.Float64),
            pl.col(f"tiempo_{name}").cast(pl.Float64),
        )
    )


def _get_label(name: str) -> tuple[str, str]:
    """Devuelve (nombre_visible, estrategia) para un resultado."""
    try:
        prog_name = extract_program_name(name)
        strat = cargar_programa(prog_name).estrategia
    except Exception:
        strat = ""
    col = strat or name
    label = f"{name} [{strat}]" if strat else name
    return col, label


def compare_results(name_a: str, name_b: str, tol: float) -> dict:
    """Compara dos resultados y devuelve métricas, DataFrame y etiquetas."""
    df_a = load_result(name_a)
    df_b = load_result(name_b)

    # Renombrar columnas para que no colisionen después del join
    df_a = df_a.rename(
        {f"perdida_{name_a}": "perdida_a", f"tiempo_{name_a}": "tiempo_a"}
    )
    df_b = df_b.rename(
        {f"perdida_{name_b}": "perdida_b", f"tiempo_{name_b}": "tiempo_b"}
    )

    df = (
        df_a.join(df_b, on="indice", how="inner")
        .with_columns((pl.col("perdida_a") - pl.col("perdida_b")).abs().alias("diff"))
        .with_columns(
            (pl.col("diff") / pl.col("perdida_b").abs().clip(lower_bound=1e-12)).alias(
                "rel_diff"
            ),
            (pl.col("diff") <= tol).alias("ok"),
        )
    )

    col_a, label_a = _get_label(name_a)
    col_b, label_b = _get_label(name_b)

    return {
        "df": df,
        "n": len(df),
        "n_ok": int(df["ok"].sum()),
        "max_d": float(df["diff"].max()),
        "mean_d": float(df["diff"].mean()),
        "median_d": float(df["diff"].median()),
        "col_a": col_a,
        "col_b": col_b,
        "label_a": label_a,
        "label_b": label_b,
    }


def parse_result_key(name: str) -> tuple[str, str, str]:
    """'program-01/N10B--phi--patron-2' → (dataset, estrategia, patron)."""
    stem = name.split("/", 1)[1] if "/" in name else name
    parts = stem.split("--", 2)
    if len(parts) != 3:
        return ("", "", "")
    return parts[0], parts[1], parts[2]


def group_selected(names: list[str]) -> dict[tuple[str, str], list[str]]:
    """Agrupa nombres por (dataset, patron) → {(dataset,patron): [nombre, ...]}."""
    groups: dict[tuple[str, str], list[str]] = {}
    for name in names:
        dataset, _, patron = parse_result_key(name)
        key = (dataset, patron)
        groups.setdefault(key, []).append(name)
    return groups


def compare_group_n(names: list[str], tol: float) -> dict:
    """Compara N resultados (mismo dataset+patron). Retorna merged df + stats por par."""
    loaded: dict[str, pl.DataFrame] = {}
    for name in names:
        _, estrategia, _ = parse_result_key(name)
        try:
            df = (
                pl.read_csv(RESULTADOS_DIR / f"{name}.csv", infer_schema_length=0)
                .select(["indice", "perdida", "tiempo_wall_s"])
                .with_columns(
                    pl.col("indice").cast(pl.Int64),
                    pl.col("perdida").cast(pl.Float64).alias(estrategia),
                    pl.col("tiempo_wall_s").cast(pl.Float64).alias(f"{estrategia}_t"),
                )
                .drop("perdida", "tiempo_wall_s")
            )
        except Exception:
            continue  # CSV vacío o corrupto
        if df.is_empty():
            continue
        loaded[estrategia] = df

    strategies = sorted(loaded.keys())
    merged = loaded[strategies[0]]
    for strat in strategies[1:]:
        merged = merged.join(loaded[strat], on="indice", how="inner")

    pairs: dict[tuple[str, str], dict] = {}
    for a, b in combinations(strategies, 2):
        diff = (pl.col(a) - pl.col(b)).abs()
        pair_df = merged.with_columns(diff.alias("_diff")).with_columns(
            (pl.col("_diff") <= tol).alias("_ok")
        )
        n = len(pair_df)
        n_ok = int(pair_df["_ok"].sum())
        pairs[(a, b)] = {
            "n": n,
            "n_ok": n_ok,
            "max_d": float(pair_df["_diff"].max()),
            "mean_d": float(pair_df["_diff"].mean()),
            "median_d": float(pair_df["_diff"].median()),
        }

    return {"merged": merged, "strategies": strategies, "pairs": pairs}


def build_rich_table_n(pairs: dict, strategies: list[str], tol: float) -> Table:
    """Tabla Rich con stats pairwise para N estrategias."""
    tabla = Table(
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        box=rich_box.SIMPLE,
        row_styles=["", "on grey11"],
        padding=(0, 1),
    )
    tabla.add_column("par", no_wrap=True)
    tabla.add_column("n_ok / n", no_wrap=True)
    tabla.add_column("%", no_wrap=True)
    tabla.add_column("max_diff", no_wrap=True)
    tabla.add_column("mean_diff", no_wrap=True)

    for (a, b), s in pairs.items():
        n, n_ok = s["n"], s["n_ok"]
        pct = 100 * n_ok / n if n > 0 else 0.0
        ok_str = f"[green]{n_ok}/{n}[/green]" if n_ok == n else f"[yellow]{n_ok}/{n}[/yellow]"
        tabla.add_row(
            f"{a} vs {b}",
            ok_str,
            f"{pct:.1f}%",
            f"{s['max_d']:.6f}",
            f"{s['mean_d']:.6f}",
        )
    return tabla


def build_rich_table(df: pl.DataFrame, col_a: str, col_b: str) -> Table:
    """Construye la tabla Rich con los resultados de la comparación."""
    tabla = Table(
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        box=rich_box.SIMPLE,
        row_styles=["", "on grey11"],
        padding=(0, 1),
    )
    tabla.add_column("idx", no_wrap=True)
    tabla.add_column(col_a, no_wrap=True)
    tabla.add_column(col_b, no_wrap=True)
    tabla.add_column("diff", no_wrap=True)
    tabla.add_column("rel %", no_wrap=True)
    tabla.add_column("✓", no_wrap=True, justify="center")

    for row in df.iter_rows(named=True):
        ok = row["ok"]
        diff_str = (
            f"[green]{row['diff']:.6f}[/green]"
            if ok
            else f"[red]{row['diff']:.6f}[/red]"
        )
        tabla.add_row(
            str(row["indice"]),
            f"{row['perdida_a']:.6f}",
            f"{row['perdida_b']:.6f}",
            diff_str,
            f"{row['rel_diff'] * 100:.2f}%",
            "[green]✓[/green]" if ok else "[red]✗[/red]",
        )

    return tabla
