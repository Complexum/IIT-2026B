"""Estrategia AnalyticDelta: MIP exacto via Zeta Transform sobre datos normalizados.

Normalización SIGNADA upfront: δ_i := H_i - p_i (el pivote queda en 0).
La Zeta Transform corre sobre δ (sumas pueden ser negativas) y el valor absoluto
se aplica UNA vez al final, tras promediar:

    |S_h^δ(i,A) / 2^|A|| = |mean_A(H_i) - p_i|   = costo EMD por hipercubo.

Esto evita restar el pivote en cada máscara (se resta una sola vez) y preserva la
cancelación de signo que la EMD explota — a diferencia de |H-p| (abs antes de
promediar), que sobreestima salvo cuando las caras no cruzan el pivote.

Complejidad: O(D·N·2^D).
"""

import time

import numpy as np

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.zeta import zeta_caras
from src.iit.core.system import System
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.sia import SIA


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

    Compatibilidad: mantiene la firma y las coordenadas lógicas para las variantes
    `analytic_*`. Paga un transpose y una permutación final. La ruta rápida es
    `src.iit.strategies.python.zeta.zeta_caras`, que arranca de `NCube.data` (ya plano) y no
    permuta nada — usarla en código nuevo.
    """
    from src.iit.strategies.python.zeta import zeta_inplace

    pivot_flat = sum((pivot_idx[j] & 1) << j for j in range(D))
    # data_nd tiene eje d+1 = dims[d]; el plano con bit j = dims[j] es el orden F.
    flat = data_nd.reshape((N,) + (2,) * D).transpose(0, *range(D, 0, -1))
    flat = np.ascontiguousarray(flat.reshape(N, 1 << D), dtype=np.float32)
    zeta_inplace(flat, N, D, pivot_flat)
    # Permutación final para devolver coordenadas lógicas (compatibilidad).
    return flat[:, pivot_flat ^ np.arange(1 << D, dtype=np.int64)]


class Analytic(SIA, nombre="analytic"):
    """MIP exacto: Zeta Transform O(D·N·2^D) + evaluación 100% vectorizada."""

    def winner(self, sistema: System) -> tuple[tuple[int, ...], tuple[int, ...]]:
        D = len(sistema.dims)
        N = len(sistema.ncubos)

        # Ruta rápida: δ = H − p y Zeta en un solo paso, sin gather (ver zeta.py).
        sumas, pivot_flat = zeta_caras(sistema)

        # C_conc = |mean_full(δ)| = |mean_full(H) - p| para colapso.
        # sumas[:, pivot_flat ^ full_mask] es la suma sobre TODAS las celdas.
        all_mean = sumas[:, pivot_flat ^ ((1 << D) - 1)] / float(1 << D) if D else None
        conc_costs = np.abs(all_mean)
        conc_idx = int(np.argmin(conc_costs))
        C_conc = float(conc_costs[conc_idx])

        if D <= 1:
            return (sistema.ncubos[conc_idx].indice,), ()

        full_mask = (1 << D) - 1

        all_m = np.arange(1, (full_mask >> 1) + 1, dtype=np.int32)
        all_c = (full_mask ^ all_m).astype(np.int32)

        bits2 = (all_m[:, None] >> np.arange(D, dtype=np.int32)) & 1
        sz_A = bits2.sum(axis=1).astype(np.float32)
        sz_B = float(D) - sz_A

        sum_A = sumas[:, pivot_flat ^ all_m.astype(np.int64)]
        sum_B = sumas[:, pivot_flat ^ all_c.astype(np.int64)]
        val_A = sum_A / (2.0**sz_A)  # mean_A(δ) = mean_A(H) - p
        val_B = sum_B / (2.0**sz_B)  # mean_B(δ) = mean_B(H) - p

        # C_dist = Σ_i min(|mean_A(δ_i)|, |mean_B(δ_i)|) — abs tras promediar
        f_all = np.minimum(np.abs(val_A), np.abs(val_B)).sum(axis=0)

        best_idx = int(np.argmin(f_all))
        best_val = float(f_all[best_idx])
        best_mask_a = int(all_m[best_idx])
        best_mask_b = int(all_c[best_idx])

        # Condición de colapso vs distribución (sin factor 2, como analytic.py)
        if C_conc <= best_val:
            return (sistema.ncubos[conc_idx].indice,), ()

        def __derive(mask_a: int) -> tuple[tuple, tuple]:
            mask_b = full_mask ^ mask_a
            sz_a = bin(mask_a).count("1")
            sz_b = D - sz_a
            va = sumas[:, pivot_flat ^ mask_a] / (1 << sz_a)
            vb = sumas[:, pivot_flat ^ mask_b] / (1 << sz_b)
            # Hipercubo va a A si |mean_B(δ)| <= |mean_A(δ)| (pivote ya en 0)
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
