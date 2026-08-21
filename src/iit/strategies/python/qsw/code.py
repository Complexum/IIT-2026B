"""Estrategia QSW: MIP via Stoer-Wagner × Queyranne sobre el oráculo Zeta.

Híbrido de las dos rutas que ya viven en este repo:

- de **Queyranne** (`qn`): el conjunto base ``V = dims ∪ indices`` (D+N vértices,
  la bipartición fiel de las dos capas temporales) y el **oráculo Zeta exacto**
  ``qn.oracle.f_cara`` — O(N) por corte, sin remarginalizar.
- de **Stoer-Wagner**: la matriz densa de pesos y el MAO con **keys
  incrementales**, que no consulta el oráculo durante el ordenamiento.

La cota de Queyranne es O(V³) consultas; el `memoria_bipart` de `qn` colapsa las
repetidas y en la práctica evalúa ~1.3·V² cortes distintos. **La ganancia real de
QSW no es el conteo sino la forma**: `qn` paga un round-trip Python→numpy por
consulta (V=30 → ~1.2k round-trips), QSW agrupa las mismas lecturas en O(V)
batches (29) y `modo=estatico` en 2. Medido con TPMs sintéticas, la búsqueda de QSW
es prácticamente plana mientras el resto crece::

    V     zeta (compartido)   búsqueda qsw   TOT analytic   TOT qn   TOT qsw
    32              0.0098s        0.0022s        0.0135s  0.0267s  0.0108s
    36              0.0467s        0.0019s        0.0611s  0.0633s  0.0420s
    40              0.2849s        0.0023s        0.3361s  0.2159s  0.1885s
    44              1.7642s        0.0028s        2.4364s  2.0888s  1.2417s

Es decir: a partir de V≈40 el costo de QSW **es** el precómputo Zeta —inherente a
los datos, O(D·N·2^D), compartido con `analytic` y `qn`— y la búsqueda deja de
contar. `analytic` en cambio sigue enumerando 2^(D−1) máscaras.

La función objetivo es simétrica y anclada (``f(∅) = f(V) = 0``, ``f(S) = f(V∖S)``),
condición que Stoer-Wagner y Queyranne exigen — ver ``QSW.md`` para la derivación.

Opciones (atributos, declarados en ``opciones`` y validados por
``SIA.aplicar_opciones``; se setean con ``ejecutar(..., opciones={...})`` o desde
el CLI con ``cli run execution X --opcion modo=estatico``):

``modo``
  - ``exacto`` (default) — tras cada contracción recalcula con el oráculo la fila
    del supernodo. Sin drift del surrogate.
  - ``estatico`` — ``W[s] += W[t]`` de Stoer-Wagner puro. Cero consultas tras el
    seed, así que las O(V²) evaluaciones caben en un único batch upfront (2 batches
    totales vs 29) → paralelizable por completo. A cambio el error de segundo orden
    se acumula por contracción.

``backend``
  - ``python`` (default) — la ruta numpy de ``core.py``.
  - ``c`` — búsqueda en ``libqsw.so`` (ver ``backend.py``). Falla explícito si no
    está compilada; nunca degrada a Python en silencio.
  - ``auto`` — C si la librería está disponible, Python si no.

Módulos (una responsabilidad cada uno):
  ``core.py`` algoritmo · ``backend.py`` selección/carga del kernel · ``code.py``
  glue con SIA · ``reference.py`` port crudo de Stoer-Wagner.
"""

import time

import numpy as np

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.qn.oracle import f_cara_batch, preparar_oraculo
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.python.qsw.backend import candidatas_c, resolver_backend
from src.iit.strategies.python.qsw.core import (
    OraculoCache,
    puntuar_candidatas,
    stoer_wagner_queyranne,
)


class QSW(SIA, nombre="qsw"):
    """MIP via Stoer-Wagner sobre el oráculo Zeta de Queyranne: O(V²) consultas."""

    #: Opciones configurables. El primer valor de cada tupla es el default.
    #: `ejecutar(..., opciones={"backend": "c"})` las setea tras validarlas.
    opciones = {
        "modo": ("exacto", "estatico"),
        "backend": ("python", "c", "auto"),
    }

    modo: str = "exacto"
    backend: str = "python"

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

        backend = resolver_backend(self.backend)
        if backend == "c":
            # El kernel C sólo BUSCA: devuelve las candidatas de fase. La decisión
            # final se toma acá con la f exacta, igual que en la ruta Python.
            vert_kind = np.array([0] * D + [1] * N, dtype=np.int32)
            vert_slot = np.array(list(range(D)) + list(range(N)), dtype=np.int32)
            candidatas = candidatas_c(
                oraculo.sumas, N, D, V, vert_kind, vert_slot, self.modo
            )
            candidatas += [1 << u for u in range(V)]  # pre-pass de singletons
            f = OraculoCache(f_batch)
            valor, mask = puntuar_candidatas(V, f, candidatas)
            self._oraculo_stats = f
        else:
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
