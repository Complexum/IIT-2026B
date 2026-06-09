"""Comando: results <nombre> [query SQL].

Carga un CSV de resultados y permite ejecutar queries SQL sobre él
usando Polars. Si no se proporciona query, muestra todas las filas.
"""

from pathlib import Path

import polars as pl

from src.cli.utils import console, error, print_table, success, warn
from src.io.manager import RESULTADOS_DIR, listar_resultados


def _resolver_ruta_resultado(name: str) -> Path | None:
    """Dado un nombre de programa o ruta completa, retornar la ruta del CSV.

    - Si contiene '/', asume ruta completa relativa a data/output/.
    - Si no, busca resultados que empiecen con ese nombre de programa.
    """
    if "/" in name:
        ruta = RESULTADOS_DIR / f"{name}.csv"
        if ruta.exists():
            return ruta
        return None

    # Buscar resultados que empiecen con name/
    coincidencias = [r for r in listar_resultados() if r.startswith(f"{name}/")]
    if not coincidencias:
        return None
    if len(coincidencias) > 1:
        warn(f"Múltiples resultados para '{name}'. Usando el primero.")
        for c in coincidencias:
            console.print(f"  - {c}")
    return RESULTADOS_DIR / f"{coincidencias[0]}.csv"


def handle(args) -> None:
    name = args.name
    query = args.query

    ruta = _resolver_ruta_resultado(name)
    if ruta is None:
        error(f"No se encontró resultado para '{name}'.")
        return

    try:
        df = pl.read_csv(ruta)
    except Exception as e:
        error(f"Error al leer CSV: {e}")
        return

    if query:
        try:
            # Polars SQL usa "self" para referirse al DataFrame
            if "FROM" not in query.upper():
                query = f"{query} FROM self"
            df = df.sql(query)
        except Exception as e:
            error(f"Error en query SQL: {e}")
            return

    if df.is_empty():
        warn("La query no retornó resultados.")
        return

    # Convertir a formato para print_table
    columnas = df.columns
    filas = df.rows()

    total = len(filas)
    titulo = f"Resultado: {name}"
    if query:
        titulo += f" | Query: {query[:50]}{'...' if len(query) > 50 else ''}"

    print_table(columnas, filas, title=titulo)
    success(f"{total} fila(s) mostrada(s).")
