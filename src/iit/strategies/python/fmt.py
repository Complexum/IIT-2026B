"""Formateo de biparticiones para la estrategia Basic."""

from src.iit.base.consts import COLON_DELIM, VOID_STR
from src.iit.strategies.python.func import ABECEDARY, Parte, fmt_particion


def _letras(indices: tuple[int, ...]) -> list[str]:
    return [ABECEDARY[i] for i in indices]


def __join(letras: list[str]) -> str:
    return COLON_DELIM.join(letras) if letras else VOID_STR


def __repartir(
    rep: tuple[int, ...],
    part: tuple[int, ...],
) -> tuple[list[str], list[str]]:
    """Reparte los índices del repertorio en parte primaria y dual."""
    prim, dual = [], []
    for i in rep:
        (prim if i in part else dual).append(ABECEDARY[i])
    return prim, dual


def fmt_parts(
    particion: tuple[tuple[int, ...], tuple[int, ...]],
    completo: tuple[tuple[int, ...], tuple[int, ...]],
) -> str:
    """Formatea una bi-partición (alcance, mecanismo) como dos columnas de brackets."""
    alcance_rep, mecanismo_rep = completo
    alcance_part, mecanismo_part = particion

    purv_prim, purv_dual = __repartir(alcance_rep, alcance_part)
    mech_prim, mech_dual = __repartir(mecanismo_rep, mecanismo_part)

    return fmt_particion(
        Parte(__join(purv_prim), __join(mech_prim)),
        Parte(__join(purv_dual), __join(mech_dual)),
    )
