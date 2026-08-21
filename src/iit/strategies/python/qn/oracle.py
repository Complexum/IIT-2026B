"""Oráculo Zeta compartido por las variantes de QNodes (qn, qn_mul, qn_mpi, qn_cuda).

Precomputa ``sumas[i, mask]`` (transformada Zeta sobre δ = H − p, ver ``oracle.md``) y
expone ``f_cara`` para leer el EMD de cualquier corte ``(alcance, mecanismo)`` en O(N).
Mismo dtype/orden de operaciones que la ruta serial original: todas las variantes que
usan este módulo para *rankear* candidatos durante el MAO obtienen lecturas
bit-idénticas a ``qn``.
"""

from dataclasses import dataclass, field

import numpy as np

from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import hyperfaces


@dataclass
class Oraculo:
    sumas: np.ndarray  # (N, 2^D) float32, Zeta transform de δ = H - p
    pos_dim: dict[int, int]  # dim -> posición de bit en la máscara
    indices_order: np.ndarray  # (N,) int64, orden de sistema.ncubos
    full_mask: int
    D: int
    pos_idx: dict[int, int] = field(default_factory=dict)  # índice de ncubo -> fila


def preparar_oraculo(sistema: System) -> Oraculo:
    """Precomputa ``sumas[i, m]`` (Zeta sobre δ = H − p) y los mapas auxiliares."""
    dims = sistema.dims
    D = len(dims)
    full_mask = (1 << D) - 1
    pos_dim = {d: i for i, d in enumerate(dims)}
    indices_order = np.fromiter((c.indice for c in sistema.ncubos), dtype=np.int64)

    data_nd = np.stack([c.ndata for c in sistema.ncubos])
    N = data_nd.shape[0]
    pivot_idx = tuple(int(sistema.estado_inicial[d]) for d in dims)
    pivot_vals = data_nd[(slice(None),) + pivot_idx]  # (N,)
    # Normalización firmada: δ = H − p (pivote queda en 0).
    delta_nd = data_nd - pivot_vals.reshape((N,) + (1,) * D)
    sumas = hyperfaces(N, D, delta_nd, pivot_idx)

    return Oraculo(
        sumas=sumas,
        pos_dim=pos_dim,
        indices_order=indices_order,
        full_mask=full_mask,
        D=D,
        pos_idx={int(idx): i for i, idx in enumerate(indices_order)},
    )


def f_cara(
    oraculo: Oraculo, alcance: tuple[int, ...], mecanismo: tuple[int, ...]
) -> float:
    """EMD del corte ``(alcance, mecanismo)`` leído de las sumas precomputadas.

    Reproduce ``emd_efecto(bipartir(alc, mec).distribucion_marginal(), ρ)``:
    cada cubo aporta |mean_{complemento(mec)}(δ)| si está en ``alcance``,
    o |mean_{mec}(δ)| en caso contrario.
    """
    m = 0
    for d in mecanismo:
        m |= 1 << oraculo.pos_dim[d]
    cmask = oraculo.full_mask ^ m
    sz_a = bin(m).count("1")

    val_a = np.abs(oraculo.sumas[:, m]) / (1 << sz_a)  # |mean_mec(δ)|
    val_b = np.abs(oraculo.sumas[:, cmask]) / (1 << (oraculo.D - sz_a))  # |mean_compl(δ)|

    if alcance:
        in_alc = np.zeros(oraculo.indices_order.shape[0], dtype=bool)
        pos_idx = oraculo.pos_idx
        for idx in alcance:
            in_alc[pos_idx[idx]] = True
        cost = np.where(in_alc, val_b, val_a)
    else:
        cost = val_a
    return float(cost.sum())


def f_cara_batch(
    oraculo: Oraculo,
    masks_mec: np.ndarray,
    alc_bool: np.ndarray,
) -> np.ndarray:
    """Versión batch de ``f_cara``: evalúa K cortes en una sola operación numpy.

    Args:
        masks_mec: (K,) int, máscara de *mecanismo* de cada corte (bits = ``pos_dim``).
        alc_bool:  (K, N) bool, ``alc_bool[k, i]`` = el ncubo en la fila ``i`` está
                   en el alcance del corte ``k``.

    Returns:
        (K,) float32 — mismo valor que ``f_cara`` corte a corte (bit-idéntico: mismo
        dtype y mismo orden de operaciones).

    Las O(V²) consultas que Stoer-Wagner/Queyranne piden se sirven en 1 llamada,
    eliminando el overhead de Python que domina el costo real del MAO.
    """
    m = np.asarray(masks_mec, dtype=np.int64)
    c = np.int64(oraculo.full_mask) ^ m

    # popcount por máscara → divisores 2^|A| como float32 (evita upcast a float64)
    sz_a = np.fromiter((int(x).bit_count() for x in m), dtype=np.int32, count=m.size)
    den_a = np.exp2(sz_a.astype(np.float32), dtype=np.float32)
    den_b = np.exp2((oraculo.D - sz_a).astype(np.float32), dtype=np.float32)

    val_a = np.abs(oraculo.sumas[:, m]) / den_a  # (N, K) = |mean_mec(δ)|
    val_b = np.abs(oraculo.sumas[:, c]) / den_b  # (N, K) = |mean_compl(δ)|

    cost = np.where(np.asarray(alc_bool, dtype=bool).T, val_b, val_a)
    return cost.sum(axis=0)
