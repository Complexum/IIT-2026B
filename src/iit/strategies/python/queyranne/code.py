"""Estrategia Queyranne: MIP via minimización de función submodular simétrica.

Algoritmo de Queyranne (Math. Prog. 1998): dada una función simétrica $f: 2^V → R$, encuentra la bipartición de mínimo valor en $O(|V|^3)$ evaluaciones oracle, reemplazando la búsqueda exhaustiva $O(2^{(D-1)})$ de analytic.

Complejidad: $O(D^3\times N) vs O(2^{(D-1)}·N)$ de analytic.
El oracle se evalúa de forma lazy (con cache): solo $O(D^3)$ masks distintos
en lugar de los $2^D$ del precómputo completo. Speedup teórico ~$2^D/D^3$.

Correctitud: garantizada si f es submodular. Empíricamente ~97-100% exacta
para funciones análogas no-submodulares (Kitazono et al., Entropy 2018).
Singleton pre-pass cubre el caso más común de falla no-submodular.
"""

import time
from typing import Callable

import numpy as np

from src.iit.strategies.python.fmt import fmt_parts
from src.iit.base.funcs import emd_efecto
from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.core.solution import Solution
from src.iit.core.system import System
from src.iit.strategies.python.sia import SIA
# from src.infra.middlewares.profile import perfilar


def oracle(
    N: int,
    D: int,
    data_nd: np.ndarray,
    pivot_idx: tuple[int, ...],
    pivot_vals: np.ndarray,
    full_mask: int,
) -> tuple[Callable[[int], float], Callable[[int], tuple]]:
    r"""Crea oracle lazy con cache para f(mask_a) y means(mask_a).

    $f(\text{mask}_a) = \sum_{i} \min(|\text{mean}_{B(i)} - \text{pivot}_i|, |\text{mean}_{A(i)} - \text{pivot}_i|)$
      mean_A = promedio sobre A-dims libres (B fijo en pivot) → costo de NO estar en alcance
      mean_B = promedio sobre B-dims libres (A fijo en pivot) → costo de estar en alcance

    Evalúa solo los masks pedidos durante el MAO (O(D³) únicos).
    """
    _means_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def __compute_means(mask_a: int) -> tuple[np.ndarray, np.ndarray]:
        if mask_a in _means_cache:
            return _means_cache[mask_a]
        mask_b = full_mask ^ mask_a
        # mean_A: dims en mask_a libres, resto fijo en pivot
        slc_a = tuple(
            [slice(None)]
            + [slice(None) if (mask_a >> d) & 1 else pivot_idx[d] for d in range(D)]
        )
        # mean_B: dims en mask_b libres, resto fijo en pivot
        slc_b = tuple(
            [slice(None)]
            + [slice(None) if (mask_b >> d) & 1 else pivot_idx[d] for d in range(D)]
        )
        mean_a = data_nd[tuple(slc_a)].reshape(N, -1).mean(axis=1)
        mean_b = data_nd[tuple(slc_b)].reshape(N, -1).mean(axis=1)
        _means_cache[mask_a] = (mean_a, mean_b)
        _means_cache[mask_b] = (mean_b, mean_a)  # complemento gratis
        return mean_a, mean_b

    def f(mask_a: int) -> float:
        if mask_a == 0 or mask_a == full_mask:
            return 0.0
        mean_a, mean_b = __compute_means(mask_a)
        cost_alcance = np.abs(mean_b - pivot_vals)
        cost_no_alcance = np.abs(mean_a - pivot_vals)
        return float(np.minimum(cost_alcance, cost_no_alcance).sum())

    def means(mask_a: int) -> tuple[np.ndarray, np.ndarray]:
        return __compute_means(mask_a)

    return f, means


def queyranne(D: int, f: Callable[[int], float], full_mask: int) -> tuple[float, int]:
    """Minimiza f(mask_a) via Maximum Adjacency Ordering (Queyranne 1998).

    Pre-pass: verifica singletons (f({d}) para cada d). Cubre el caso más
    común de falla en funciones no-submodulares: O(D) calls adicionales.

    Loop principal: D-1 iteraciones de MAO, O(D²) calls por iteración → O(D³) total.

    Returns:
        (best_val, best_mask_a): valor mínimo y máscara del lado ganador.
    """
    V: list[int] = [1 << d for d in range(D)]

    # Pre-pass: singletons (salvaguarda para funciones no-submodulares)
    best_val = float("inf")
    best_mask_a = V[0]
    for singleton in V:
        val = f(singleton)
        if val < best_val:
            best_val = val
            best_mask_a = singleton

    while len(V) > 1:
        # --- Maximum Adjacency Ordering ---
        # Regla de Queyranne (legal ordering):
        #
        #     v_{i+1} = argmin_v [ f(A ∪ {v}) − f({v}) ]
        #
        # El término −f({v}) no es decorativo y el sentido tampoco: sobre una f de
        # corte de grafo, f(A ∪ v) = f(A) + f(v) − 2·w(A,v), de modo que la ganancia
        # vale f(A) − 2·w(A,v). Como f(A) es constante durante el argmin, minimizarla
        # equivale a maximizar la adyacencia w(A,v) de Stoer-Wagner. Sin el −f({v}), o
        # maximizando en vez de minimizar, se elige otro vértice y se pierde la
        # garantía de par colgante que hace exacto al algoritmo sobre funciones
        # submodulares.
        #
        # `f_solo[v] = f({v})` se calcula una vez: no depende de A.
        remaining = list(V)
        a_mask = 0
        f_solo: dict[int, float] = {v: f(v) for v in remaining}
        key: dict[int, float] = dict(f_solo)
        order: list[tuple[int, float]] = []

        while remaining:
            u = min(remaining, key=lambda v: key[v] - f_solo[v])  # noqa: B023
            curr_key = key[u]
            remaining.remove(u)
            order.append((u, curr_key))
            a_mask |= u
            for w in remaining:
                key[w] = f(a_mask | w)

        # Par colgante: pendant_val = s_key = f(V \ {t}) = f({t}) por simetría
        s_node, s_key = order[-2]
        t_node, _t_key = order[-1]
        pendant_val = s_key

        if pendant_val < best_val:
            best_val = pendant_val
            best_mask_a = t_node

        # Contraer s y t en un supernodo
        V.remove(s_node)
        V.remove(t_node)
        V.append(s_node | t_node)

    return best_val, best_mask_a


# @perfilar
class Queyranne(SIA, nombre="queyranne"):
    """MIP via Queyranne: O(D³·N) con oracle lazy vs O(2^(D-1)·N) de analytic."""

    def winner(self, sistema: System) -> tuple[tuple[int, ...], tuple[int, ...]]:
        D = len(sistema.dims)
        N = len(sistema.ncubos)

        data_nd = np.stack([c.ndata for c in sistema.ncubos])
        pivot_idx = tuple(int(sistema.estado_inicial[dim]) for dim in sistema.dims)
        pivot_vals = data_nd[(slice(None),) + pivot_idx]  # shape (N,)

        # Baseline de concentración
        all_mean = data_nd.reshape(N, -1).mean(axis=1)
        conc_costs = np.abs(all_mean - pivot_vals)
        conc_idx = int(np.argmin(conc_costs))
        C_conc = float(conc_costs[conc_idx])

        if D <= 1:
            return (sistema.ncubos[conc_idx].indice,), ()

        full_mask = (1 << D) - 1

        # Oracle lazy con cache — solo evalúa los O(D³) masks pedidos por MAO
        f, means = oracle(N, D, data_nd, pivot_idx, pivot_vals, full_mask)

        best_val, best_mask_a = queyranne(D, f, full_mask)

        if C_conc <= best_val:
            return (sistema.ncubos[conc_idx].indice,), ()

        # Derivar alcance: nodo i va a alcance si cost_alcance ≤ cost_no_alcance
        # cost_alcance = |mean_b - pivot|,  cost_no_alcance = |mean_a - pivot|
        mean_a, mean_b = means(best_mask_a)
        node_in_a = np.abs(mean_b - pivot_vals) <= np.abs(mean_a - pivot_vals)

        alcance = tuple(c.indice for i, c in enumerate(sistema.ncubos) if node_in_a[i])
        mecanismo = tuple(sistema.dims[d] for d in range(D) if (best_mask_a >> d) & 1)
        return alcance, mecanismo

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
        texto_particion = fmt_parts(
            (alcance, mecanismo), (self.sistema.indices, self.sistema.dims)
        )

        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=float(perdida) if perdida != INFTY_POS else FLOAT_ZERO,
            distribucion_subsistema=dm_original,
            distribucion_particion=dm,
            particion=texto_particion.strip(),
            tiempo_total=tiempo,
            quiere_hablar=False,
        )
