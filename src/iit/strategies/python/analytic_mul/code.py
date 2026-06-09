"""Estrategia AnalyticMUL: Analytic (signed-δ) con evaluación de máscaras en multiprocessing.

Misma estrategia óptima que `analytic` (normalización SIGNADA δ = H − p, pivote→0, valor
absoluto tras promediar). La fase de evaluación de las M = 2^(D-1) máscaras se reparte entre
procesos con `multiprocessing.Pool`; el driver (Zeta transform, argmin, derive, EMD real) queda
secuencial. Espejo CPU multi-core de `analytic_concurrent` (ThreadPool) y `qn_mul`.

Cada worker lee `sumas` (poblado por el initializer del Pool) y calcula
`f(m) = Σ_i min(|sumas[i,m]/2^a|, |sumas[i,m^c]/2^(D-a)|)` sobre su chunk → resultado **idéntico**
a `analytic`. `multiprocessing` es stdlib (sin extra). Con `fork` (macos/linux) `sumas` se comparte
copy-on-write; con `spawn` se serializa una vez por worker.
"""

import multiprocessing as mp
import os
import time

import numpy as np

from src.iit.base.app import aplicacion
from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import Analytic, hyperfaces
from src.iit.strategies.python.fmt import fmt_parts

MIN_PARALLEL_MASKS = 64  # D≥8 → overhead de IPC amortizado

# ── Estado del worker (poblado por el initializer del Pool) ────────────────
_W_SUMAS: np.ndarray | None = None
_W_FULL: int = 0
_W_D: int = 0


def _init_worker(sumas: np.ndarray, full_mask: int, D: int) -> None:
    global _W_SUMAS, _W_FULL, _W_D
    _W_SUMAS = sumas
    _W_FULL = full_mask
    _W_D = D


def _eval_chunk_mp(chunk_m: np.ndarray) -> np.ndarray:
    """Worker: f(m) para un chunk de máscaras sobre δ-sumas globales."""
    sumas = _W_SUMAS
    D = _W_D
    chunk_c = (_W_FULL ^ chunk_m).astype(np.int32)
    bits2 = (chunk_m[:, None] >> np.arange(D, dtype=np.int32)) & 1
    sz_A = bits2.sum(axis=1).astype(np.float32)
    sz_B = float(D) - sz_A
    sA = sumas[:, chunk_m] / (2.0**sz_A)  # mean_A(δ)
    sB = sumas[:, chunk_c] / (2.0**sz_B)  # mean_B(δ)
    return np.minimum(np.abs(sA), np.abs(sB)).sum(axis=0)


class AnalyticMUL(Analytic, nombre="analytic_mul"):
    """Analytic con evaluación de máscaras en paralelo via multiprocessing.Pool."""

    def __init__(self, subsistema: System, n_workers: int | None = None) -> None:
        super().__init__(subsistema)
        self.n_workers = n_workers or max(1, os.cpu_count() or 1)

    def winner(self, sistema: System) -> tuple[tuple[int, ...], tuple[int, ...]]:
        D = len(sistema.dims)
        N = len(sistema.ncubos)

        data_nd = np.stack([c.ndata for c in sistema.ncubos])
        pivot_idx = tuple(int(sistema.estado_inicial[dim]) for dim in sistema.dims)
        pivot_vals = data_nd[(slice(None),) + pivot_idx]  # (N,)

        # Normalización SIGNADA: δ = H − p, el pivote queda en 0.
        delta_nd = data_nd - pivot_vals.reshape((N,) + (1,) * D)

        all_mean = delta_nd.reshape(N, -1).mean(axis=1)
        conc_costs = np.abs(all_mean)
        conc_idx = int(np.argmin(conc_costs))
        C_conc = float(conc_costs[conc_idx])

        if D <= 1:
            return (sistema.ncubos[conc_idx].indice,), ()

        full_mask = (1 << D) - 1
        all_m = np.arange(1, (full_mask >> 1) + 1, dtype=np.int32)
        M = len(all_m)

        # Umbral: caer al winner serial óptimo si hay pocos masks o un solo worker
        if M < MIN_PARALLEL_MASKS or self.n_workers <= 1:
            return super().winner(sistema)

        sumas = hyperfaces(N, D, delta_nd, pivot_idx)

        n_workers = min(self.n_workers, M)
        chunk_size = max(1, (M + n_workers - 1) // n_workers)
        chunks = [all_m[i : i + chunk_size] for i in range(0, M, chunk_size)]

        platform = "fork" if aplicacion.op_system == "macos" else "spawn"
        ctx = mp.get_context(platform)
        with ctx.Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(sumas, full_mask, D),
        ) as pool:
            f_all = np.concatenate(pool.map(_eval_chunk_mp, chunks))

        best_idx = int(np.argmin(f_all))
        best_val = float(f_all[best_idx])
        best_mask_a = int(all_m[best_idx])
        best_mask_b = int(full_mask ^ best_mask_a)

        if C_conc <= best_val:
            return (sistema.ncubos[conc_idx].indice,), ()

        def __derive(mask_a: int) -> tuple[tuple, tuple]:
            mask_b = full_mask ^ mask_a
            sz_a = bin(mask_a).count("1")
            sz_b = D - sz_a
            va = sumas[:, mask_a] / (1 << sz_a)
            vb = sumas[:, mask_b] / (1 << sz_b)
            in_a = np.abs(vb) <= np.abs(va)
            alc = tuple(c.indice for i, c in enumerate(sistema.ncubos) if in_a[i])
            mec = tuple(sistema.dims[d] for d in range(D) if (mask_a >> d) & 1)
            return alc, mec

        alc_a, mec_a = __derive(best_mask_a)
        alc_b, mec_b = __derive(best_mask_b)

        dm_orig = sistema.distribucion_marginal()
        emd_a = emd_efecto(
            sistema.bipartir(alc_a, mec_a).distribucion_marginal(), dm_orig
        )
        emd_b = emd_efecto(
            sistema.bipartir(alc_b, mec_b).distribucion_marginal(), dm_orig
        )

        return (alc_a, mec_a) if emd_a <= emd_b else (alc_b, mec_b)

    def resolver(self) -> Solution:
        dm_original = self.distribucion
        t0 = time.perf_counter()

        if not self.sistema.indices or not self.sistema.dims:
            return Solution(
                estrategia=self.nombre.capitalize(),
                perdida=FLOAT_ZERO,
                distribucion_subsistema=dm_original,
                distribucion_particion=dm_original,
                particion=fmt_parts(((), ()), ((), ())).strip(),
                tiempo_total=time.perf_counter() - t0,
                quiere_hablar=False,
            )

        alcance, mecanismo = self.winner(self.sistema)
        particion_sistema = self.sistema.bipartir(alcance, mecanismo)
        dm = particion_sistema.distribucion_marginal()
        perdida = emd_efecto(dm, dm_original)

        tiempo = time.perf_counter() - t0
        texto = fmt_parts(
            (alcance, mecanismo), (self.sistema.indices, self.sistema.dims)
        )

        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=float(perdida) if perdida != INFTY_POS else FLOAT_ZERO,
            distribucion_subsistema=dm_original,
            distribucion_particion=dm,
            particion=texto.strip(),
            tiempo_total=tiempo,
            quiere_hablar=False,
        )
