"""Estrategia AnalyticMPI: Analytic (signed-δ) con evaluación de máscaras distribuida en MPI.

Misma estrategia óptima que `analytic` (normalización SIGNADA δ = H − p, pivote→0, valor
absoluto tras promediar). La fase de evaluación de las M = 2^(D-1) máscaras se reparte entre
ranks MPI con `mpi4py.futures.MPIPoolExecutor`; el driver (Zeta transform, argmin, derive, EMD
real) corre secuencial en rank 0. Espejo distribuido de `analytic_mul` y `qn_mpi`.

Cada worker lee `sumas` (poblado por el initializer del executor) y calcula
`f(m) = Σ_i min(|sumas[i,m]/2^a|, |sumas[i,m^c]/2^(D-a)|)` sobre su chunk → resultado **idéntico**
a `analytic`.

Ejecución (cluster):
    uv pip install -e ".[mpi]"
    mpiexec -n <P> python -m mpi4py.futures -m src.cli run execution <name>

Hard-fail: sin `mpi4py` instalado o `comm.size < 2`, `resolver` lanza `RuntimeError` (la
estrategia sigue registrada y visible en CLI). Nota: `sumas (N, 2^D)` se difunde a cada rank vía
initializer (one-time); para D grande es el cuello de botella de red.
"""

import time

import numpy as np

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import Analytic, hyperfaces
from src.iit.strategies.python.fmt import fmt_parts

MIN_PARALLEL_MASKS = 64  # D≥8 → overhead de red amortizado

# ── Estado del worker (poblado por el initializer del executor) ────────────
_W_SUMAS: np.ndarray | None = None
_W_FULL: int = 0
_W_D: int = 0


def _init_worker(sumas: np.ndarray, full_mask: int, D: int) -> None:
    global _W_SUMAS, _W_FULL, _W_D
    _W_SUMAS = sumas
    _W_FULL = full_mask
    _W_D = D


def _eval_chunk_mpi(chunk_m: np.ndarray) -> np.ndarray:
    """Worker MPI: f(m) para un chunk de máscaras sobre δ-sumas globales."""
    return _eval_chunk(chunk_m, _W_SUMAS, _W_FULL, _W_D)


def _eval_chunk(
    chunk_m: np.ndarray, sumas: np.ndarray, full_mask: int, D: int
) -> np.ndarray:
    chunk_c = (full_mask ^ chunk_m).astype(np.int32)
    bits2 = (chunk_m[:, None] >> np.arange(D, dtype=np.int32)) & 1
    sz_A = bits2.sum(axis=1).astype(np.float32)
    sz_B = float(D) - sz_A
    sA = sumas[:, chunk_m] / (2.0**sz_A)  # mean_A(δ)
    sB = sumas[:, chunk_c] / (2.0**sz_B)  # mean_B(δ)
    return np.minimum(np.abs(sA), np.abs(sB)).sum(axis=0)


class AnalyticMPI(Analytic, nombre="analytic_mpi"):
    """Analytic con evaluación de máscaras distribuida en ranks MPI (mpi4py)."""

    def winner(
        self, sistema: System, executor, n_workers: int
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
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

        sumas = hyperfaces(N, D, delta_nd, pivot_idx)

        # Umbral: evaluar inline si hay pocos masks (la red no compensa)
        if M < MIN_PARALLEL_MASKS:
            f_all = _eval_chunk(all_m, sumas, full_mask, D)
        else:
            # Difunde el estado a cada rank (initializer-equivalente) antes del map.
            list(
                executor.map(
                    _init_worker,
                    [sumas] * n_workers,
                    [full_mask] * n_workers,
                    [D] * n_workers,
                )
            )
            n_chunks = min(n_workers, M)
            chunk_size = max(1, (M + n_chunks - 1) // n_chunks)
            chunks = [all_m[i : i + chunk_size] for i in range(0, M, chunk_size)]
            f_all = np.concatenate(list(executor.map(_eval_chunk_mpi, chunks)))

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
        try:
            from mpi4py import MPI
            from mpi4py.futures import MPIPoolExecutor
        except ImportError as e:
            raise RuntimeError(
                "analytic_mpi requiere mpi4py. Instala '.[mpi]' y lanza con: "
                "mpiexec -n <P> python -m mpi4py.futures -m src.cli run execution <name>"
            ) from e

        comm = MPI.COMM_WORLD
        if comm.size < 2:
            raise RuntimeError(
                "analytic_mpi requiere ejecutarse con mpiexec (mínimo 2 workers). "
                "Comando: mpiexec -n <P> python -m mpi4py.futures -m src.cli run execution <name>"
            )

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

        n_workers = max(1, comm.size - 1)
        with MPIPoolExecutor() as executor:
            alcance, mecanismo = self.winner(self.sistema, executor, n_workers)

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
