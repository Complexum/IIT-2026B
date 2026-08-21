"""Utilidades CSV para el tab Execution."""

import csv
from pathlib import Path

# `tiempo_wall_s` y las columnas de recursos miden **sólo el algoritmo**: el
# monitor arranca después de preparar el subsistema. La preparación cuesta lo
# mismo para todas las estrategias, así que incluirla comprimía todos los
# speedups hacia 1 (escondía más de la mitad de la ventaja de `qsw+backend=c`).
# `tiempo_preparacion_s` guarda ese costo por separado.
#
# Los CSV anteriores a este cambio no traen `tiempo_preparacion_s`, y en ellos
# `tiempo_wall_s` sí incluye la preparación: `compare` avisa al mezclarlos.
CSV_HEADERS = [
    "indice",
    "estado",
    "condicion",
    "alcance",
    "mecanismo",
    "perdida",
    "tiempo_wall_s",
    "tiempo_preparacion_s",
    "tiempo_cpu_s",
    "cpu_user_s",
    "cpu_sys_s",
    "mem_rss_mb",
    "gpu_mem_mb",
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
