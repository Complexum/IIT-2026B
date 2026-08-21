"""Lógica de comparación de resultados CSV usando polars."""

from itertools import combinations
from pathlib import Path

import polars as pl
from rich import box as rich_box
from rich.table import Table

from src.io.manager import listar_resultados
from src.tui.run.helpers import cargar_programa, extract_program_name

RESULTADOS_DIR = Path("data/output")

# Marca del formato nuevo de CSV. En los anteriores `tiempo_wall_s` incluía la
# preparación del subsistema; en los nuevos mide sólo el algoritmo. Comparar
# tiempos entre ambos es peras con manzanas (los `perdida` sí son comparables).
COL_FORMATO_NUEVO = "tiempo_preparacion_s"


def formato_nuevo(name: str) -> bool:
    """True si el CSV mide el tiempo del algoritmo sin la preparación."""
    try:
        cabecera = pl.read_csv(
            RESULTADOS_DIR / f"{name}.csv", n_rows=0, infer_schema_length=0
        ).columns
    except Exception:
        return False
    return COL_FORMATO_NUEVO in cabecera


def formatos_mezclados(names: list[str]) -> list[str]:
    """Nombres en formato viejo, si el conjunto mezcla ambos. Lista vacía si no."""
    viejos = [n for n in names if not formato_nuevo(n)]
    return viejos if viejos and len(viejos) < len(names) else []


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


def times_summary(merged: pl.DataFrame, strategies: list[str]) -> dict[str, dict]:
    """Resumen de cinco números (+ bigotes y media) del tiempo de cada estrategia.

    `compare_group_n` responde *si dos estrategias coinciden*; esto responde
    **cuál conviene**. Un promedio solo no alcanza: los barridos mezclan
    combinaciones con D chico y D grande, así que la distribución es asimétrica y
    la media queda arrastrada por la cola. La mediana y los cuartiles dicen qué
    pasa en el caso típico, y los bigotes dónde están los casos malos.

    Bigotes al estilo Tukey: el dato más extremo que todavía cae dentro de
    `Q1 − 1.5·IQR` / `Q3 + 1.5·IQR`. Se reporta el dato real y no la cota, que es
    lo que dibuja un boxplot; lo que queda afuera se cuenta como atípico.
    """
    resumen: dict[str, dict] = {}
    for estrategia in strategies:
        col = f"{estrategia}_t"
        if col not in merged.columns:
            continue
        serie = merged[col].drop_nulls().drop_nans().sort()
        if serie.is_empty():
            continue

        q1 = float(serie.quantile(0.25, interpolation="linear"))
        q3 = float(serie.quantile(0.75, interpolation="linear"))
        iqr = q3 - q1
        cota_inf, cota_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        dentro = serie.filter((serie >= cota_inf) & (serie <= cota_sup))
        # Si el IQR es 0 (todas iguales) `dentro` puede quedar vacío: caer al dato.
        bigote_inf = float(dentro.min()) if not dentro.is_empty() else float(serie.min())
        bigote_sup = float(dentro.max()) if not dentro.is_empty() else float(serie.max())

        resumen[estrategia] = {
            "n": len(serie),
            "min": float(serie.min()),
            "bigote_inf": bigote_inf,
            "q1": q1,
            "mediana": float(serie.median()),
            "media": float(serie.mean()),
            "q3": q3,
            "bigote_sup": bigote_sup,
            "max": float(serie.max()),
            "atipicos": int(len(serie) - len(dentro)),
            "total": float(serie.sum()),
        }
    return resumen


def build_rich_table_tiempos(resumen: dict[str, dict]) -> Table:
    """Resumen de tiempos, una columna por estrategia y la mejor primero.

    Va transpuesta (métricas en filas) a propósito: con 12 métricas y el formato
    normal la tabla pasa de 120 caracteres y la terminal la trunca justo en las
    columnas que importan. Así entra en 80 y además se compara leyendo a lo largo
    de una fila, que es la pregunta real ("¿cuál conviene?").

    En ms: `tiempo_wall_s` da valores del orden de 1e-3 y en segundos la tabla
    queda llena de ceros.
    """
    orden = sorted(resumen.items(), key=lambda kv: kv[1]["mediana"])

    tabla = Table(
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        box=rich_box.SIMPLE,
        padding=(0, 1),
        title="tiempo por combinación (ms)",
        title_style="bold",
    )
    tabla.add_column("", no_wrap=True, style="bold")
    for i, (estrategia, _) in enumerate(orden):
        tabla.add_column(
            estrategia, justify="right", no_wrap=True,
            style="green" if i == 0 else "",
        )

    if not orden:
        return tabla

    filas = [
        ("n", "n", None),
        ("max", "max", "ms"),
        ("bigote ↑", "bigote_sup", "ms"),
        ("Q3", "q3", "ms"),
        ("media", "media", "ms"),
        ("mediana", "mediana", "ms"),
        ("Q1", "q1", "ms"),
        ("bigote ↓", "bigote_inf", "ms"),
        ("min", "min", "ms"),
        ("atípicos", "atipicos", None),
        ("total", "total", "ms"),
    ]
    # Orden de arriba hacia abajo como se dibuja un boxplot: max arriba, min abajo.
    for etiqueta, clave, unidad in filas:
        tabla.add_row(
            etiqueta,
            *(
                f"{s[clave] * 1e3:.3f}" if unidad == "ms" else str(s[clave])
                for _, s in orden
            ),
        )

    mejor = orden[0][1]["mediana"]
    tabla.add_row(
        "vs mejor",
        *(
            "—" if i == 0
            else (f"{s['mediana'] / mejor:.2f}x" if mejor > 0 else "—")
            for i, (_, s) in enumerate(orden)
        ),
    )
    return tabla


def stamps_veredict(resumen: dict[str, dict]) -> str:
    """Una línea de conclusión: quién gana, por mediana y por tiempo total.

    Los dos números se reportan siempre, incluso cuando gana la misma estrategia,
    porque la brecha entre ellos *es* el dato. Sobre N15A/patron-2, `qsw` va 1.49x
    por mediana y 2.15x por total: la ventaja no está en la combinación típica sino
    en las caras, que es justo lo que un promedio solo escondería. Cuando además
    los ganadores difieren, una estrategia gana en el caso común y la otra en la
    cola — y eso hay que decirlo, no promediarlo.
    """
    if len(resumen) < 2:
        return ""

    def _factor(clave: str) -> tuple[str, float]:
        orden = sorted(resumen.items(), key=lambda kv: kv[1][clave])
        mejor, segunda = orden[0], orden[1]
        razon = segunda[1][clave] / mejor[1][clave] if mejor[1][clave] > 0 else 1.0
        return mejor[0], razon

    gana_mediana, f_mediana = _factor("mediana")
    gana_total, f_total = _factor("total")

    if gana_mediana == gana_total:
        return (
            f"Más rápida: [green]{gana_mediana}[/green] — "
            f"{f_mediana:.2f}x por mediana, {f_total:.2f}x por tiempo total "
            f"(sobre la siguiente)"
        )
    return (
        f"Mediana: gana [green]{gana_mediana}[/green] ({f_mediana:.2f}x) · "
        f"total: gana [green]{gana_total}[/green] ({f_total:.2f}x) — "
        f"una es mejor en el caso típico y la otra en la cola"
    )


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
