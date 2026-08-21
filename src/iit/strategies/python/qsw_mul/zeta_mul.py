"""Transformada Zeta repartida entre procesos: las N filas son independientes.

Única responsabilidad de este módulo. Es la mitad paralela de `qsw_mul`; la
estrategia vive en `code.py`.

Por qué acá y no en el MAO
--------------------------
Medido, el precómputo Zeta **es** la corrida: a n=22 son 0.957 s de los 0.946 s
totales de `qsw`, y la búsqueda 2.8 ms (`QSW.md` §6). Paralelizar el MAO no
movería la aguja; el Zeta sí. Y `zeta_inplace` recorre `(N, 2^D)` una vez por
dimensión sin cruzar filas —cada fila es un butterfly independiente— así que el
reparto natural es por filas (`QSW.md` §8).

Por qué memoria compartida y no `Pool.map` de arrays
----------------------------------------------------
El arreglo pesa 80 MB a D=20 y 336 MB a D=22. Devolver los chunks por `pool.map`
los picklea de ida y de vuelta: dos pasadas extra sobre el arreglo más la
serialización, comparable al costo total del Zeta en un algoritmo que ya está
limitado por ancho de banda. Y `fork` por sí solo tampoco alcanza: copy-on-write
hace que las escrituras del worker sean privadas y se pierdan al salir.

Se usa `RawArray` (`multiprocessing.sharedctypes`) y no `shared_memory` porque
`SharedMemory` registra cada attach en el `resource_tracker`, y evitar sus avisos
de "leaked shared_memory objects" bajo `spawn` exige el kwarg `track=` (Python
3.13+) o tocar API privada — y el proyecto declara `requires-python = ">=3.12"`.
`RawArray` da la misma memoria sin copia, se hereda bajo `fork`, viaja por
duplicación de descriptor bajo `spawn`, y lo libera el GC.

Lo que queda serial
-------------------
El llenado y la resta del pivote: 2 pasadas contra las D del butterfly, o sea una
fracción serial de 2/(D+2) ≈ 9 % a D=20 (techo de Amdahl ~11×). A cambio, llenar
directo en el buffer compartido evita el `np.stack` + `ascontiguousarray` de
`zeta_caras`, que hoy toca un pico de 12·N·2^D bytes (float64 y float32 vivas a la
vez); acá el pico es 4·N·2^D más una fila float64.
"""

import multiprocessing as mp
import numpy as np

from src.iit.base.app import aplicacion
from src.iit.strategies.python.qn.oracle import Oraculo
from src.iit.strategies.python.zeta import pivote_plano, zeta_inplace

# ── Estado del worker (poblado por el initializer del Pool) ────────────────
_W_VISTA: np.ndarray | None = None
_W_D: int = 0
_W_PIVOT: int = 0


def _init_worker(buf, N: int, D: int, pivot_flat: int) -> None:
    """Mapea el buffer compartido como `(N, 2^D)` float32. Cero copias."""
    global _W_VISTA, _W_D, _W_PIVOT
    _W_VISTA = np.frombuffer(buf, dtype=np.float32).reshape(N, 1 << D)
    _W_D = D
    _W_PIVOT = pivot_flat


def _zeta_filas(rango: tuple[int, int]) -> None:
    """Butterfly in-place sobre `vista[lo:hi]`. No devuelve nada: escribe en la
    memoria compartida, que es justamente el punto."""
    lo, hi = rango
    # Una rebanada de filas de un (N, 2^D) C-contiguo es contigua, así que
    # `zeta_inplace` se reusa tal cual — el butterfly no se duplica.
    zeta_inplace(_W_VISTA[lo:hi], hi - lo, _W_D, _W_PIVOT)


def repartir(N: int, n_workers: int) -> list[tuple[int, int]]:
    """Rebanadas contiguas de filas, lo más parejas posible.

    Con N=20 y 16 workers quedan 4 rebanadas de 2 y 12 de 1: el techo efectivo es
    ~10×, no 16×. El reparto por filas está acotado por N.
    """
    n = min(n_workers, N)
    base, resto = divmod(N, n)
    cortes = []
    lo = 0
    for i in range(n):
        hi = lo + base + (1 if i < resto else 0)
        cortes.append((lo, hi))
        lo = hi
    return cortes


def preparar_oraculo_mul(sistema, n_workers: int) -> Oraculo:
    """`preparar_oraculo` con el butterfly repartido entre procesos.

    Devuelve un `Oraculo` idéntico al de la ruta secuencial —mismo dtype, mismo
    orden de sumas dentro de cada fila— así que `f_cara_batch` y toda la búsqueda
    quedan **bit-idénticas** a `qsw`.
    """
    ncubos = sistema.ncubos
    N, D = len(ncubos), len(sistema.dims)
    total = 1 << D

    plataforma = "fork" if aplicacion.op_system == "macos" else "spawn"
    ctx = mp.get_context(plataforma)

    buf = ctx.RawArray("f", N * total)
    vista = np.frombuffer(buf, dtype=np.float32).reshape(N, total)

    # δ = H − p con el pivote en 0: se aplica por fila mientras se llena, para no
    # recorrer el arreglo una segunda vez.
    pivot_flat = pivote_plano(sistema)
    for i, c in enumerate(ncubos):
        fila = vista[i]
        fila[:] = c.data
        fila -= fila[pivot_flat]

    rangos = repartir(N, n_workers)
    with ctx.Pool(
        processes=len(rangos),
        initializer=_init_worker,
        initargs=(buf, N, D, pivot_flat),
    ) as pool:
        pool.map(_zeta_filas, rangos)

    indices_order = np.fromiter((c.indice for c in ncubos), dtype=np.int64)
    return Oraculo(
        sumas=vista,
        pos_dim={d: i for i, d in enumerate(sistema.dims)},
        indices_order=indices_order,
        full_mask=total - 1,
        D=D,
        pos_idx={int(idx): i for i, idx in enumerate(indices_order)},
        pivot_flat=pivot_flat,
    )
