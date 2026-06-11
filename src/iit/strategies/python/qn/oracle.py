"""Oráculo Zeta compartido por las variantes de QNodes (qn, qn_mul, qn_mpi, qn_cuda).

Precomputa ``sumas[i, mask]`` (transformada Zeta sobre δ = H − p, ver ``oracle.md``) y
expone ``f_cara`` para leer el EMD de cualquier corte ``(alcance, mecanismo)`` en O(N).
Mismo dtype/orden de operaciones que la ruta serial original: todas las variantes que
usan este módulo para *rankear* candidatos durante el MAO obtienen lecturas
bit-idénticas a ``qn``.
"""

from dataclasses import dataclass

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
        in_alc = np.isin(
            oraculo.indices_order, np.fromiter(alcance, dtype=np.int64)
        )
        cost = np.where(in_alc, val_b, val_a)
    else:
        cost = val_a
    return float(cost.sum())
