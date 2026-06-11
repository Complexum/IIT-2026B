"""Generación de redes (TPMs) aleatorias."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.iit.base.app import aplicacion
from src.iit.base.consts import CSV_EXTENSION

REDES_DIR = Path("data/input/networks")

_optimizar_executor = ThreadPoolExecutor(max_workers=1)


def generar_red(dimensiones: int, datos_deterministas: bool = True, optimizar: bool = True) -> str:
    """Generar una red (TPM) en notación little-endian y guardarla como CSV.

    Args:
        dimensiones: Número de nodos en la red
        datos_deterministas: True → binario (0/1), False → probabilidades [0,1]
        optimizar: Si True, genera el sidecar .npy en background tras el CSV.

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

    nombre_base = f"N{dimensiones}{sufijo}"
    nombre = f"{nombre_base}.{CSV_EXTENSION}"
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

    if optimizar:
        from src.io.manager import optimizar_red
        _optimizar_executor.submit(optimizar_red, nombre_base)

    return nombre


def peso_estimado(dimensiones: int) -> float:
    """Estimar el tamaño del archivo en GB para N dimensiones.

    El archivo CSV contiene2^N filas × N valores float (8 bytes c/u).
    Factor empírico ~9× debido al formato CSV (comas, newlines, precisión).
    """
    num_estados = 1 << dimensiones
    return (num_estados * dimensiones * 9) / (1024**3)


# Alias
estimate_size = peso_estimado
