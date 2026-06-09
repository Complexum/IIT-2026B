"""Helpers del tab Dataset: parseo de inputs y formateo de sistemas."""

from src.iit.core.system import System


def parsear_estado(s: str) -> tuple[int, ...] | None:
    """Parsea un string binario a valores de estado.

    '101' → (1, 0, 1)
    Retorna None si es vacío o contiene caracteres no-binarios.
    """
    s = s.strip()
    if not s or not all(c in "01" for c in s):
        return None
    return tuple(int(c) for c in s)


def parsear_indices(s: str) -> tuple[int, ...]:
    """Parsea un string binario a índices de dimensión donde bit = 0.

    '101' → (0, 2)  — dimensiones 0 y 2 activas
    '001' → (2,)    — solo dimensión 2

    Retorna () si es vacío o inválido.
    """
    s = s.strip()
    if not s or not all(c in "01" for c in s):
        return ()
    return tuple(i for i, c in enumerate(s) if c == "0")


def formatear_sistema(sistema: System) -> str:
    """Formatea un System para mostrar en las columnas scrolleables."""
    if not sistema.ncubos:
        return "(vacío — sin ncubos)"

    lineas: list[str] = []
    lineas.append(f"Indices: {sistema.indices}")
    lineas.append(f"Dims:    {sistema.dims}")
    lineas.append(f"Estado:  {sistema.estado_inicial}")
    lineas.append("")

    for c in sistema.ncubos:
        lineas.append(str(c))
        lineas.append("")

    lineas.append("Distribución marginal:")
    dist = sistema.distribucion_marginal()
    lineas.append(f"  [{', '.join(f'{v:.6f}' for v in dist)}]")
    return "\n".join(lineas)
