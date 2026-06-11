"""Carga y agrupación de resultados desde data/output/ para figuras de paper."""

from pathlib import Path

import polars as pl


RESULTADOS_DIR = Path("data/output")


def parse_stem(name: str) -> tuple[str, str, str]:
    """'program-01/N10B--phi--patron-2' → (dataset, estrategia, patron)."""
    stem = name.split("/", 1)[1] if "/" in name else name
    parts = stem.split("--", 2)
    if len(parts) != 3:
        raise ValueError(f"Nombre de resultado inesperado: '{name}'")
    return parts[0], parts[1], parts[2]


def group_selected(names: list[str]) -> dict[tuple[str, str], dict[str, str]]:
    """Agrupa nombres de resultado por (dataset, patron).

    Retorna {(dataset, patron): {estrategia: nombre_completo}}.
    """
    groups: dict[tuple[str, str], dict[str, str]] = {}
    for name in names:
        dataset, estrategia, patron = parse_stem(name)
        key = (dataset, patron)
        groups.setdefault(key, {})[estrategia] = name
    return groups


def load_group_df(
    group: dict[str, str],
    output_dir: Path = RESULTADOS_DIR,
) -> pl.DataFrame:
    """Carga y fusiona CSVs de un grupo (mismo dataset+patron) en un DataFrame.

    Columnas resultantes: indice, alcance, mecanismo, subsys, {strat}, {strat}_t, ...
    """
    dfs: list[pl.DataFrame] = []
    base: pl.DataFrame | None = None

    for estrategia, name in sorted(group.items()):
        path = output_dir / f"{name}.csv"
        try:
            df = (
                pl.read_csv(path, infer_schema_length=0)
                .select(["indice", "alcance", "mecanismo", "perdida", "tiempo_wall_s"])
                .with_columns(
                    pl.col("indice").cast(pl.Int64),
                    pl.col("perdida").cast(pl.Float64),
                    pl.col("tiempo_wall_s").cast(pl.Float64),
                )
            )
        except Exception:
            continue
        if df.is_empty():
            continue

        df = df.rename({"perdida": estrategia, "tiempo_wall_s": f"{estrategia}_t"})

        if base is None:
            base = df.select(["indice", "alcance", "mecanismo"]).with_columns(
                (pl.col("alcance").str.len_chars() + pl.col("mecanismo").str.len_chars()).alias("subsys")
            )

        dfs.append(df.drop("alcance", "mecanismo"))

    if not dfs or base is None:
        return pl.DataFrame()

    merged = base
    for df in dfs:
        merged = merged.join(df, on="indice", how="inner")

    return merged.sort("subsys")


def load_all_groups(
    names: list[str],
    output_dir: Path = RESULTADOS_DIR,
) -> dict[tuple[str, str], pl.DataFrame]:
    """Carga todos los grupos a partir de los nombres seleccionados en la TUI."""
    groups = group_selected(names)
    result = {}
    for key, strat_map in groups.items():
        df = load_group_df(strat_map, output_dir)
        if not df.is_empty():
            result[key] = df
    return result