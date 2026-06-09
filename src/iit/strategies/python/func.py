from dataclasses import dataclass

import numpy as np

from src.iit.base.consts import ABC_START, VOID_STR


def get_labels(n: int) -> tuple[str, ...]:
    """
    Genera etiquetas en formato Excel para n nodos (A, B, …, Z, AA, AB, …).
    """

    def get_excel_column(n: int) -> str:
        return (
            ""
            if n <= 0
            else get_excel_column((n - 1) // 26) + chr((n - 1) % 26 + ord(ABC_START))
        )

    return tuple(get_excel_column(i) for i in range(1, n + 1))


ABECEDARY = get_labels(40)
LOWER_ABECEDARY = [letter.lower() for letter in ABECEDARY]


# ── Formateo de particiones ───────────────────────────────────────────────────


@dataclass
class Parte:
    """Una columna de una partición: numerador (purview) y denominador (mecanismo)."""

    numerador: str
    denominador: str


def lil_endian(n: int) -> np.ndarray:
    """Índices de permutación big-endian → little-endian para 2^n estados.

    pyphi almacena repertorios en little-endian; esta permutación convierte
    el orden de estados para visualización consistente con el resto del sistema.
    """
    size = 1 << n
    return np.array(
        [int(f"{i:0{n}b}"[::-1], 2) for i in range(size)], dtype=np.intp
    )


def fmt_biparticion_fuerza_bruta(
    dual: list,
    prim: list,
) -> str:
    """Formatea una bipartición de pyphi (mecanismo/purview por part) como brackets.

    dual: [dual_mechanism, dual_purview]  — indices de nodos (tuplas pyphi)
    prim: [prim_mechanism, prim_purview]
    """
    dual_mech, dual_purv = dual
    prim_mech, prim_purv = prim

    def _fmt(indices) -> str:
        return ",".join(ABECEDARY[i] for i in indices) if indices else VOID_STR

    return fmt_particion(
        Parte(_fmt(dual_purv), _fmt(dual_mech)),
        Parte(_fmt(prim_purv), _fmt(prim_mech)),
    )


def fmt_particion(*partes: Parte) -> str:
    """Formatea N partes de una k-partición como columnas de brackets adyacentes.

    Cada parte produce una columna con la forma:
        ⎛ numerador  ⎞
        ⎝ denominador⎠

    Todas las columnas se concatenan en una sola línea superior y una inferior.

    Ejemplos
    --------
    Bi-partición (A | B,C) con (∅ | A,B,C):
        >>> fmt_particion(Parte("A", "∅"), Parte("B,C", "A,B,C"))
        '⎛ A ⎞⎛ B,C  ⎞\\n⎝ ∅ ⎠⎝ A,B,C⎠'
    """

    def _col(p: Parte) -> tuple[str, str]:
        num = p.numerador or VOID_STR
        den = p.denominador or VOID_STR
        w = max(len(num), len(den)) + 2
        return f"⎛{num:^{w}}⎞", f"⎝{den:^{w}}⎠"

    tops, bottoms = zip(*(_col(p) for p in partes))
    return "".join(tops) + "\n" + "".join(bottoms)
