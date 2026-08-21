"""Transformada Zeta sobre las caras del hipercubo — el precómputo del oráculo.

Única responsabilidad de este módulo. Lo comparten `analytic` y la familia
Queyranne (`qn`, `qn_mul`, `qn_mpi`, `qn_cuda`, `qsw`).

Por qué existe
--------------
Medido, este precómputo **es** el costo de las estrategias que lo usan: a n=22
son 0.957 s de los 0.946 s totales de `qsw` (la búsqueda son 2.8 ms). Y corre a
17–19 GB/s, es decir saturando el ancho de banda de DRAM: está limitado por
memoria, no por cómputo. Cada pasada de más sobre el arreglo cuesta.

La versión anterior (`analytic.hyperfaces`) gastaba el **95 %** del tiempo en un
*gather* que además asignaba dos arreglos `(2^D, D) int32` — 738 MB a D=22 sólo
en índices:

    bits   = (u_idx[:, None] >> arange(D)) & 1     # (2^D, D)
    states = array(pivot_idx) ^ bits               # (2^D, D)
    sumas  = data_nd[(slice(None),) + tuple(states[:, d] ...)]

Ese gather es sólo una permutación XOR con el pivote, y se puede plegar en la
**dirección** del butterfly. Además `NCube.data` ya es plano con bit *j* =
`dims[j]` (ver `ncube.py`), así que tampoco hace falta `ndata` ni el reshape.

Convención de coordenadas
-------------------------
`zeta_caras` devuelve el arreglo en **coordenadas delta**: la suma de la cara
lógica `m` vive en la posición `pivot_flat ^ m`. Esto evita una permutación final
de 2^D elementos. Usar `leer(sumas, pivot_flat, m)` o indexar directamente con
`sumas[:, pivot_flat ^ m]` — el XOR sobre un vector de K máscaras es despreciable.
"""

import numpy as np


def pivote_plano(sistema) -> int:
    """Índice plano del estado pivote: bit *j* = `dims[j]`, igual que `NCube.data`."""
    return sum(
        (int(sistema.estado_inicial[d]) & 1) << j for j, d in enumerate(sistema.dims)
    )


def zeta_inplace(flat: np.ndarray, N: int, D: int, pivot_flat: int) -> np.ndarray:
    """Butterfly de suma-sobre-subconjuntos **in-place**, con la dirección del pivote.

    Recurrencia Zeta estándar: `Z[m | 2^d] += Z[m \\ 2^d]`. Acá el arreglo está en
    coordenadas delta (posición = `pivot_flat ^ m`), así que el destino de cada
    suma es el lado *opuesto* al bit del pivote — de ahí el swap de `(s, o)`.

    D pasadas sobre el arreglo; no asigna nada.
    """
    for d in range(D):
        t = flat.reshape(N, 1 << (D - d - 1), 2, 1 << d)
        fuente, destino = (0, 1) if not ((pivot_flat >> d) & 1) else (1, 0)
        t[:, :, destino, :] += t[:, :, fuente, :]
    return flat


def zeta_caras(sistema, dtype=np.float32, kernel=None) -> tuple[np.ndarray, int]:
    """Ruta rápida: de `System` a las sumas de cara, sin gather ni temporales.

    Args:
        kernel: butterfly alternativo con la firma `(flat, N, D, pivot_flat)`. Por
            defecto `zeta_inplace` (numpy). `qsw.backend.zeta_c` usa el kernel C,
            que es bit-exacto y 5–10× más rápido.

    Returns:
        `(sumas, pivot_flat)` — `sumas` en coordenadas delta, ver el docstring del
        módulo. La suma de la cara `m` es `sumas[:, pivot_flat ^ m]`.
    """
    N, D = len(sistema.ncubos), len(sistema.dims)
    flat = np.stack([c.data for c in sistema.ncubos])
    flat = np.ascontiguousarray(flat, dtype=dtype)
    pivot_flat = pivote_plano(sistema)
    # Normalización firmada δ = H − p: el pivote queda en 0 y el valor absoluto se
    # aplica DESPUÉS de promediar, preservando la cancelación de signo que la EMD
    # explota (ver el docstring de `analytic`).
    flat -= flat[:, pivot_flat][:, None]
    (kernel or zeta_inplace)(flat, N, D, pivot_flat)
    return flat, pivot_flat


def leer(sumas: np.ndarray, pivot_flat: int, m) -> np.ndarray:
    """Suma(s) de la(s) cara(s) lógica(s) `m`. Acepta un int o un array de máscaras."""
    return sumas[:, pivot_flat ^ np.asarray(m)]
