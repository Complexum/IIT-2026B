"""Estrategia SWQ: MIP via Stoer-Wagner × Queyranne sobre el oráculo Zeta.

Híbrido de las dos rutas que ya viven en este repo:

- de **Queyranne** (`qn`): el conjunto base ``V = dims ∪ indices`` (D+N vértices,
  la bipartición fiel de las dos capas temporales) y el **oráculo Zeta exacto**
  ``qn.oracle.f_cara`` — O(N) por corte, sin remarginalizar.
- de **Stoer-Wagner**: la matriz densa de pesos y el MAO con **keys
  incrementales**, que no consulta el oráculo durante el ordenamiento.

La cota de Queyranne es O(V³) consultas; el `memoria_bipart` de `qn` colapsa las
repetidas y en la práctica evalúa ~1.3·V² cortes distintos. **La ganancia real de
SWQ no es el conteo sino la forma**: `qn` paga un round-trip Python→numpy por
consulta (V=30 → ~1.2k round-trips), SWQ agrupa las mismas lecturas en O(V)
batches (29) y `swq_static` en 2. Medido con TPMs sintéticas, la búsqueda de SWQ
es prácticamente plana mientras el resto crece::

    V     zeta (compartido)   búsqueda swq   TOT analytic   TOT qn   TOT swq
    32              0.0098s        0.0022s        0.0135s  0.0267s  0.0108s
    36              0.0467s        0.0019s        0.0611s  0.0633s  0.0420s
    40              0.2849s        0.0023s        0.3361s  0.2159s  0.1885s
    44              1.7642s        0.0028s        2.4364s  2.0888s  1.2417s

Es decir: a partir de V≈40 el costo de SWQ **es** el precómputo Zeta —inherente a
los datos, O(D·N·2^D), compartido con `analytic` y `qn`— y la búsqueda deja de
contar. `analytic` en cambio sigue enumerando 2^(D−1) máscaras.

La función objetivo es simétrica y anclada (``f(∅) = f(V) = 0``, ``f(S) = f(V∖S)``),
condición que Stoer-Wagner y Queyranne exigen — ver ``SWQ.md`` para la derivación.

Variantes registradas:
  - ``swq``        — modo *exacto*: recalcula con el oráculo la fila del supernodo
                     tras cada contracción. Sin drift del surrogate.
  - ``swq_static`` — modo *estático*: ``W[s] += W[t]`` de Stoer-Wagner puro. Todo
                     el oráculo cabe en un único batch upfront → paralelizable
                     por completo (multiprocessing / CUDA / kernel C).
"""

import time

import numpy as np

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.qn.oracle import f_cara_batch, preparar_oraculo
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.python.swq.core import stoer_wagner_queyranne


class SWQ(SIA, nombre="swq"):
    """MIP via Stoer-Wagner sobre el oráculo Zeta de Queyranne: O(V²) consultas."""

    modo: str = "exacto"

    def winner(self) -> tuple[tuple[int, ...], tuple[int, ...], float]:
        """Devuelve ``(alcance, mecanismo, valor_oraculo)`` del corte ganador.

        Codificación de vértices sobre un entero de Python de ``V = D + N`` bits::

            bit  0 .. D-1      → (ACTUAL, dims[j])     candidato a mecanismo
            bit  D .. D+N-1    → (EFFECT, indices[i])  candidato a alcance

        El mapeo máscara → ``(alcance, mecanismo)`` es shift + máscara: `qn` paga
        hoy un ``__flatten``/``sorted`` recursivo en cada consulta.
        """
        sistema = self.sistema
        dims = sistema.dims
        indices = sistema.indices
        D, N = len(dims), len(indices)
        V = D + N

        oraculo = preparar_oraculo(sistema)
        mec_mask = (1 << D) - 1
        desplaz = np.arange(N, dtype=np.int64)

        def f_batch(masks) -> np.ndarray:
            arr = np.fromiter(masks, dtype=object, count=len(masks))
            masks_mec = np.fromiter(
                (int(m) & mec_mask for m in arr), dtype=np.int64, count=arr.size
            )
            altos = np.fromiter(
                (int(m) >> D for m in arr), dtype=np.int64, count=arr.size
            )
            alc_bool = ((altos[:, None] >> desplaz) & 1).astype(bool)
            return f_cara_batch(oraculo, masks_mec, alc_bool)

        valor, mask, self._oraculo_stats = stoer_wagner_queyranne(
            V, f_batch, modo=self.modo
        )

        mecanismo = tuple(dims[j] for j in range(D) if (mask >> j) & 1)
        alcance = tuple(indices[i] for i in range(N) if (mask >> (D + i)) & 1)
        return alcance, mecanismo, valor

    def resolver(self) -> Solution:
        t0 = time.perf_counter()
        dm_original = self.distribucion

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

        alcance, mecanismo, _ = self.winner()

        # Reconstrucción REAL del ganador (igual que `qn`): una marginalización
        # de verdad + EMD real. Salvaguarda contra cualquier deriva del oráculo,
        # O(N·2^D) amortizado sobre toda la corrida.
        dm = self.sistema.bipartir(alcance, mecanismo).distribucion_marginal()
        perdida = emd_efecto(dm, dm_original)

        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=float(perdida) if perdida != INFTY_POS else FLOAT_ZERO,
            distribucion_subsistema=dm_original,
            distribucion_particion=dm,
            particion=fmt_parts(
                (alcance, mecanismo), (self.sistema.indices, self.sistema.dims)
            ).strip(),
            tiempo_total=time.perf_counter() - t0,
            quiere_hablar=False,
        )


class SWQEstatico(SWQ, nombre="swq_static"):
    """SWQ con contracción de Stoer-Wagner pura: todo el oráculo en 1 batch."""

    # modo = estatico | dinamico
    modo: str = "estatico"
