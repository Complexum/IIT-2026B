from src.iit.base.consts import COLON_DELIM, VOID_STR
from src.iit.strategies.python.func import ABECEDARY, LOWER_ABECEDARY


def fmt_biparticion(
    prim: list[tuple[int, int]],
    dual: list[tuple[int, int]],
    to_sort: bool = True,
) -> str:
    top_prim, bottom_prim = fmt_parte_media(prim, to_sort)
    top_dual, bottom_dual = fmt_parte_media(dual, to_sort)

    return f"{top_prim}{top_dual}\n{bottom_prim}{bottom_dual}\n"


def fmt_parte_media(
    parte: list[tuple[int, int]], a_ordenar: bool = True
) -> tuple[str, str]:
    """
    LLega del tipo
    [(0 1) (2 4) (3 5)]
    """
    if a_ordenar:
        # Ordenar por índice #
        parte.sort(key=lambda x: x[1])

    purv, mech = [], []
    for time, idx in parte:
        purv.append(ABECEDARY[idx]) if time else mech.append(LOWER_ABECEDARY[idx])

    str_purv = COLON_DELIM.join(purv) if purv else VOID_STR
    str_mech = COLON_DELIM.join(mech) if mech else VOID_STR
    width = max(len(str_purv), len(str_mech)) + 2

    return f"⎛{str_purv:^{width}}⎞", f"⎝{str_mech:^{width}}⎠"
