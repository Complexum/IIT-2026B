"""Estrategia Analytical: MIP exacto via Zeta Transform + evaluación vectorizada.

Dos innovaciones sobre todas las variantes previas:

1. Precomputo SOS (Suma-sobre-Subconjuntos / Zeta transform):
   Todas las 2^D sumas de hiper-caras en D pasadas numpy vectorizadas.
   O(D·N·2^D) vs O(N·3^D) del loop naïve de analytic_optimized.

2. Evaluación vectorizada total:
   Las ~2^(D-1) biparticiones se evalúan en un único bloque numpy (N, 2^D)
   sin ningún bucle Python, vs 155K iteraciones de analytic o 5800 de Queyranne.

Complejidad total: O(D·N·2^D).
Exactitud: 100% dentro del espacio de biparticiones de dimensiones.
"""

import time

import numpy as np

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.core.system import System
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.sia import SIA
# from src.infra.middlewares.profile import perfilar


def hyperfaces(
    N: int,
    D: int,
    data_nd: np.ndarray,
    pivot_idx: tuple[int, ...],
) -> np.ndarray:
    """Zeta transform: sumas de hiper-caras para todos los 2^D masks en O(D·N·2^D).

    sumas[:, m] = Σ_{u ⊆ m} data_nd[:, pivot XOR u]
                = suma sobre las 2^|m| celdas con dims libres = bits de m,
                  resto fijo en pivot_idx.

    D pasadas de reshape+slice sobre (N, 2^D) — sin bucle Python por mask.
    """
    u_idx = np.arange(1 << D, dtype=np.int32)  # (2^D,)
    bits = (u_idx[:, None] >> np.arange(D, dtype=np.int32)) & 1  # (2^D, D)
    states = np.array(pivot_idx, dtype=np.int32) ^ bits  # (2^D, D)

    # pivot_data[n, u] = data_nd[n, states[u,0], states[u,1], ..., states[u,D-1]]
    idx = (slice(None),) + tuple(states[:, d] for d in range(D))
    sumas: np.ndarray = data_nd[idx].astype(np.float32)  # (N, 2^D)

    # D pasadas SOS: para cada dim d, acumula subconjuntos que incluyen el bit d.
    # Reshape convierte sumas en vista (..., 2, ...) con el bit d en eje central.
    for d in range(D):
        sh = (N, 1 << (D - d - 1), 2, 1 << d)
        temp = sumas.reshape(sh)  # vista — no copia
        temp[:, :, 1, :] += temp[:, :, 0, :]

    return sumas


# @perfilar
class Analytical(SIA, nombre="analytical"):
    """MIP exacto: Zeta Transform O(D·N·2^D) + evaluación 100% vectorizada."""

    def winner(self, sistema: System) -> tuple[tuple[int, ...], tuple[int, ...]]:
        D = len(sistema.dims)
        N = len(sistema.ncubos)

        data_nd = np.stack([c.ndata for c in sistema.ncubos])
        pivot_idx = tuple(int(sistema.estado_inicial[dim]) for dim in sistema.dims)
        pivot_vals = data_nd[(slice(None),) + pivot_idx]  # (N,)

        # Concentración baseline
        all_mean = data_nd.reshape(N, -1).mean(axis=1)
        conc_costs = np.abs(all_mean - pivot_vals)
        conc_idx = int(np.argmin(conc_costs))
        C_conc = float(conc_costs[conc_idx])

        if D <= 1:
            return (sistema.ncubos[conc_idx].indice,), ()

        full_mask = (1 << D) - 1

        # ── 1. Precomputo SOS: todas las sumas en O(D·N·2^D) ──────────────────
        sumas = hyperfaces(N, D, data_nd, pivot_idx)  # (N, 2^D)

        # ── 2. Evaluación vectorizada de todas las biparticiones ──────────────
        # Sólo masks 1..full_mask//2 (la otra mitad tiene el mismo f por simetría).
        all_m = np.arange(1, (full_mask >> 1) + 1, dtype=np.int32)  # (M,)
        all_c = (full_mask ^ all_m).astype(np.int32)  # complementos

        # Popcount vectorizado (sin bucle Python)
        bits2 = (all_m[:, None] >> np.arange(D, dtype=np.int32)) & 1  # (M, D)
        sz_A = bits2.sum(axis=1).astype(np.float32)  # (M,)
        sz_B = float(D) - sz_A

        sum_A = sumas[:, all_m]  # (N, M)
        sum_B = sumas[:, all_c]  # (N, M)
        val_A = sum_A / (2.0**sz_A)  # (N, M)
        val_B = sum_B / (2.0**sz_B)

        # cost_alcance = |val_B - pivot|,  cost_no_alcance = |val_A - pivot|
        cost_alc = np.abs(val_B - pivot_vals[:, None])  # (N, M)
        cost_no = np.abs(val_A - pivot_vals[:, None])  # (N, M)
        f_all = np.minimum(cost_alc, cost_no).sum(axis=0)  # (M,)

        best_idx = int(np.argmin(f_all))
        best_val = float(f_all[best_idx])
        best_mask_a = int(all_m[best_idx])
        best_mask_b = int(all_c[best_idx])

        if C_conc <= best_val:
            return (sistema.ncubos[conc_idx].indice,), ()

        # ── 3. Derivar (alcance, mecanismo) para ambos candidatos ─────────────
        # f(mask) = f(complement) por simetría, pero la asignación de nodos difiere.
        # Probamos ambos y nos quedamos con el que da menor EMD real.
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
