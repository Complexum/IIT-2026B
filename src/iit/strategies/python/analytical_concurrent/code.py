import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.core.system import System
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.analytical.code import Analytical, hyperfaces
# from src.infra.middlewares.profile import perfilar

MIN_PARALLEL_MASKS = 64  # D≥8 → overhead amortizado
D_MAX_SOS = 25  # para D>25 usar modo lazy; D=25 → ~3.2GB en 24GB RAM


# @perfilar
class AnalyticalConcurrent(Analytical, nombre="analytical_concurrent"):
    """Analytic con evaluación de máscaras en paralelo via ThreadPool."""

    def __init__(self, subsistema: System, n_workers: int | None = None) -> None:
        super().__init__(subsistema)
        self.n_workers = n_workers or max(1, os.cpu_count() or 1)

    def winner(self, sistema: System) -> tuple[tuple[int, ...], tuple[int, ...]]:
        D = len(sistema.dims)
        N = len(sistema.ncubos)

        data_nd = np.stack([c.ndata for c in sistema.ncubos])
        pivot_idx = tuple(int(sistema.estado_inicial[dim]) for dim in sistema.dims)
        pivot_vals = data_nd[(slice(None),) + pivot_idx]  # (N,)

        # Colapso baseline
        all_mean = data_nd.reshape(N, -1).mean(axis=1)
        conc_costs = np.abs(all_mean - pivot_vals)
        conc_idx = int(np.argmin(conc_costs))
        C_conc = float(conc_costs[conc_idx])

        if D <= 1:
            return (sistema.ncubos[conc_idx].indice,), ()

        full_mask = (1 << D) - 1
        all_m = np.arange(1, (full_mask >> 1) + 1, dtype=np.int32)
        M = len(all_m)

        # Umbral: caer a secuencial si hay pocos masks o un solo worker
        if M < MIN_PARALLEL_MASKS or self.n_workers <= 1:
            return super().winner(sistema)

        n_workers = min(self.n_workers, M)
        chunk_size = max(1, (M + n_workers - 1) // n_workers)
        chunks = [all_m[i : i + chunk_size] for i in range(0, M, chunk_size)]

        # ── Modo SOS (D ≤ D_MAX_SOS): precomputa todo, threads leen sumas ──────
        if D <= D_MAX_SOS:
            sumas = hyperfaces(N, D, data_nd, pivot_idx)

            def task(chunk: np.ndarray) -> np.ndarray:
                return eval_chunk(chunk, sumas, pivot_vals, full_mask, D)

        # ── Modo lazy (D > D_MAX_SOS): cada chunk calcula sus propias medias ────
        else:

            def task(chunk: np.ndarray) -> np.ndarray:  # noqa: F811
                return eval_chunk_lazy(
                    chunk, data_nd, pivot_idx, pivot_vals, full_mask, D, N
                )

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            f_all = np.concatenate(list(ex.map(task, chunks)))

        best_idx = int(np.argmin(f_all))
        best_val = float(f_all[best_idx])
        best_mask_a = int(all_m[best_idx])
        best_mask_b = int(full_mask ^ best_mask_a)

        if C_conc <= best_val:
            return (sistema.ncubos[conc_idx].indice,), ()

        # Derivar (alcance, mecanismo) para ambos candidatos; elegir menor EMD real
        if D <= D_MAX_SOS:

            def __derive(mask_a: int) -> tuple[tuple, tuple]:
                mask_b = full_mask ^ mask_a
                sz_a = bin(mask_a).count("1")
                sz_b = D - sz_a
                va = sumas[:, mask_a] / (1 << sz_a)
                vb = sumas[:, mask_b] / (1 << sz_b)
                in_a = np.abs(vb - pivot_vals) <= np.abs(va - pivot_vals)
                alc = tuple(c.indice for i, c in enumerate(sistema.ncubos) if in_a[i])
                mec = tuple(sistema.dims[d] for d in range(D) if (mask_a >> d) & 1)
                return alc, mec
        else:

            def __derive(mask_a: int) -> tuple[tuple, tuple]:  # noqa: F811
                mask_b = full_mask ^ mask_a
                slc_a = tuple(
                    [slice(None)]
                    + [
                        slice(None) if (mask_a >> d) & 1 else pivot_idx[d]
                        for d in range(D)
                    ]
                )
                slc_b = tuple(
                    [slice(None)]
                    + [
                        slice(None) if (mask_b >> d) & 1 else pivot_idx[d]
                        for d in range(D)
                    ]
                )
                va = data_nd[slc_a].reshape(N, -1).mean(axis=1)
                vb = data_nd[slc_b].reshape(N, -1).mean(axis=1)
                in_a = np.abs(vb - pivot_vals) <= np.abs(va - pivot_vals)
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


def eval_chunk(
    chunk_m: np.ndarray,
    sumas: np.ndarray,
    pivot_vals: np.ndarray,
    full_mask: int,
    D: int,
) -> np.ndarray:
    """Evalúa f para un chunk de máscaras. Puro numpy → GIL liberado."""
    chunk_c = (full_mask ^ chunk_m).astype(np.int32)
    bits2 = (chunk_m[:, None] >> np.arange(D, dtype=np.int32)) & 1  # (chunk, D)
    sz_A = bits2.sum(axis=1).astype(np.float32)  # (chunk,)
    sz_B = float(D) - sz_A

    sA = sumas[:, chunk_m] / (2.0**sz_A)  # (N, chunk)
    sB = sumas[:, chunk_c] / (2.0**sz_B)  # (N, chunk)

    return np.minimum(
        np.abs(sB - pivot_vals[:, None]),
        np.abs(sA - pivot_vals[:, None]),
    ).sum(axis=0)  # (chunk,)


def eval_chunk_lazy(
    chunk_m: np.ndarray,
    data_nd: np.ndarray,
    pivot_idx: tuple[int, ...],
    pivot_vals: np.ndarray,
    full_mask: int,
    D: int,
    N: int,
) -> np.ndarray:
    """Evalúa f sin sumas precalculadas — para D > D_MAX_SOS (ahorra memoria)."""
    results = np.empty(len(chunk_m), dtype=np.float32)
    # d_range = np.arange(D, dtype=np.int32)

    for k, mask_a in enumerate(chunk_m):
        mask_b = int(full_mask ^ int(mask_a))
        # mean_A: dims en mask_a libres, resto fijo en pivot
        slc_a = tuple(
            [slice(None)]
            + [
                slice(None) if (int(mask_a) >> d) & 1 else pivot_idx[d]
                for d in range(D)
            ]
        )
        # mean_B: dims en mask_b libres, resto fijo en pivot
        slc_b = tuple(
            [slice(None)]
            + [slice(None) if (mask_b >> d) & 1 else pivot_idx[d] for d in range(D)]
        )
        mean_a = data_nd[slc_a].reshape(N, -1).mean(axis=1)
        mean_b = data_nd[slc_b].reshape(N, -1).mean(axis=1)
        results[k] = float(
            np.minimum(np.abs(mean_b - pivot_vals), np.abs(mean_a - pivot_vals)).sum()
        )
    return results
