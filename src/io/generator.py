"""Generación de redes (TPMs) aleatorias."""

from pathlib import Path

import numpy as np

from src.iit.base.app import aplicacion
from src.iit.base.consts import CSV_EXTENSION

REDES_DIR = Path("data/input/networks")


def generar_red(dimensiones: int, datos_deterministas: bool = True) -> str:
    """Generar una red (TPM) en notación little-endian y guardarla como CSV.

    Args:
        dimensiones: Número de nodos en la red
        datos_deterministas: True → binario (0/1), False → probabilidades [0,1]

    Returns:
        Nombre del archivo generado (ej: 'N3A.csv')
    """
    np.random.seed(aplicacion.semilla_numpy)

    if dimensiones < 1:
        raise ValueError("Las dimensiones deben ser positivas")

    num_estados = 1 << dimensiones
    REDES_DIR.mkdir(parents=True, exist_ok=True)

    sufijo = "A"
    while (REDES_DIR / f"N{dimensiones}{sufijo}.{CSV_EXTENSION}").exists():
        sufijo = chr(ord(sufijo) + 1)

    nombre = f"N{dimensiones}{sufijo}.{CSV_EXTENSION}"
    ruta = REDES_DIR / nombre

    if datos_deterministas:
        estados = np.random.randint(2, size=(num_estados, dimensiones), dtype=np.int8)
    else:
        estados = np.random.random(size=(num_estados, dimensiones))

    np.savetxt(
        ruta,
        estados,
        delimiter=",",
        fmt="%d" if datos_deterministas else "%.6f",
    )
    return nombre


def peso_estimado(dimensiones: int) -> float:
    """Estimar el tamaño del archivo en GB para N dimensiones."""
    num_estados = 1 << dimensiones
    return (num_estados * dimensiones) / (1024**3)


# Alias
estimate_size = peso_estimado
