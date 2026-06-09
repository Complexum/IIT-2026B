"""Utilidades CSV para el tab Execution."""

import csv
from pathlib import Path

CSV_HEADERS = [
    "indice",
    "estado",
    "condicion",
    "alcance",
    "mecanismo",
    "perdida",
    "tiempo",
    "particion",
    "plataforma",
]


def cargar_indices_completados(output_path: Path) -> set[int]:
    """Lee el CSV existente y retorna el set de índices ya calculados."""
    if not output_path.exists():
        return set()
    try:
        completados: set[int] = set()
        with output_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                completados.add(int(row["indice"]))
        return completados
    except Exception:
        return set()  # CSV corrupto → empezar de cero
