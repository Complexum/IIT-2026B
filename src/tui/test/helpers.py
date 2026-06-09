"""Persistencia y lógica de patrones de prueba.

Un patrón define listas de generadores para cada categoría
(estados, condiciones, alcances, mecanismos). Las combinaciones
son el producto cartesiano de los resultados de cada generador.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path

from src.tui.test.generadores import etiquetas, generar

PATRONES_DIR = Path("data/input/patrones")


def _natural_key(s: str) -> list:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", s)]


@dataclass
class Patron:
    """Un patrón de prueba con generadores por categoría."""

    nombre: str
    estados: list[str] = field(default_factory=list)
    condiciones: list[str] = field(default_factory=list)
    alcances: list[str] = field(default_factory=list)
    mecanismos: list[str] = field(default_factory=list)


# ── CRUD ─────────────────────────────────────────────────


def guardar_patron(patron: Patron) -> None:
    """Guardar patrón como JSON en data/input/patrones/."""
    PATRONES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = PATRONES_DIR / f"{patron.nombre}.json"
    ruta.write_text(json.dumps(asdict(patron), indent=2, ensure_ascii=False))


def cargar_patron(nombre: str) -> Patron:
    """Cargar patrón desde JSON."""
    ruta = PATRONES_DIR / f"{nombre}.json"
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return Patron(**datos)


def listar_patrones() -> list[str]:
    """Listar nombres de patrones disponibles."""
    if not PATRONES_DIR.exists():
        return []
    return sorted(
        (f.stem for f in PATRONES_DIR.glob("*.json")), key=_natural_key, reverse=True
    )


def eliminar_patron(nombre: str) -> bool:
    """Eliminar patrón. Retorna True si existía."""
    ruta = PATRONES_DIR / f"{nombre}.json"
    if ruta.exists():
        ruta.unlink()
        return True
    return False


# ── Combinaciones ────────────────────────────────────────


def total_combinaciones(patron: Patron) -> int:
    """Calcular total de combinaciones del producto cartesiano."""
    e = max(len(patron.estados), 1)
    c = max(len(patron.condiciones), 1)
    a = max(len(patron.alcances), 1)
    m = max(len(patron.mecanismos), 1)
    return e * c * a * m


def generar_combinaciones(patron: Patron, n: int) -> list[tuple[str, str, str, str]]:
    """Generar todas las combinaciones para n dimensiones.

    Retorna lista de (estado, condición, alcance, mecanismo) como strings binarios.
    """
    ests = [generar(g, n) for g in patron.estados] or [generar("todos", n)]
    conds = [generar(g, n) for g in patron.condiciones] or [generar("todos", n)]
    alcs = [generar(g, n) for g in patron.alcances] or [generar("todos", n)]
    mecs = [generar(g, n) for g in patron.mecanismos] or [generar("todos", n)]
    return list(product(ests, conds, alcs, mecs))


def formatear_preview(patron: Patron, n: int) -> str:
    """Genera texto de preview con todas las combinaciones."""
    combis = generar_combinaciones(patron, n)
    lineas: list[str] = []
    for i, (est, cond, alc, mec) in enumerate(combis, 1):
        etiq_alc = etiquetas(alc)
        etiq_mec = etiquetas(mec)
        lineas.append(
            f"{i:3d}. ({est} | {cond} | {alc} | {mec})  →  ({etiq_alc} | {etiq_mec})"
        )
    return "\n".join(lineas) if lineas else "(sin combinaciones)"


def siguiente_nombre_patron() -> str:
    """Genera el siguiente nombre disponible: patron-1, patron-2, ..."""
    existentes = set(listar_patrones())
    i = 1
    while f"patron-{i}" in existentes:
        i += 1
    return f"patron-{i}"
