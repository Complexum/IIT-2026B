"""QSW sobre la **especificación**: nunca materializa la TPM.

Misma búsqueda que `qsw` —Stoer-Wagner × Queyranne, `core.stoer_wagner_queyranne`
sin modificar— pero alimentada por el oráculo de Zeta **local**
(`strategies/python/zeta_local.py`), que corre la transformada sobre la tabla de
cada nodo (`2^kᵢ` celdas) en lugar de sobre la TPM de `N × 2^D`::

    Θ(Σᵢ kᵢ · 2^kᵢ)     en vez de     Θ(D · N · 2^D)

Es **exacto**, no muestreado: coincide con `qsw`/`analytic` hasta el piso de float32
de la referencia. La diferencia está en la entrada, no en el algoritmo.

Entrada
-------
Registra `necesita_espec=True`, así que recibe un `reduccion.SubEspec` en vez de un
`System`. `SubEspec` expone `distribucion_marginal` y `bipartir_marginal`, que es
todo lo que `SIA` y la reconstrucción final necesitan — por eso no hizo falta tocar
la clase base ni el runner.

La red tiene que tener sidecar `.spec.json`. Una TPM aleatoria densa **no** admite
especificación compacta (ver `io/identificabilidad.py`), así que esta estrategia no
es un reemplazo de `qsw`: es la ruta para redes definidas por su estructura.
"""

import time

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.qsw.core import stoer_wagner_queyranne
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.python.zeta_local import (
    f_cara_batch_local,
    preparar_oraculo_local,
)

import numpy as np


class QSWFst(SIA, nombre="qsw_fst", necesita_espec=True):
    """MIP sobre la especificación, con el oráculo de Zeta local."""

    opciones = {"modo": ("estatico", "exacto")}
    modo: str = "estatico"

    def resolver(self) -> Solution:
        t0 = time.perf_counter()
        sub = self.sistema  # SubEspec, no System
        dm_original = self.distribucion
        D, N = len(sub.dims), len(sub.indices)

        if D == 0 or N == 0:
            return Solution(
                estrategia=self.nombre.capitalize(),
                perdida=FLOAT_ZERO,
                distribucion_subsistema=dm_original,
                distribucion_particion=dm_original,
                particion=fmt_parts(((), ()), ((), ())).strip(),
                tiempo_total=time.perf_counter() - t0,
                quiere_hablar=False,
            )

        orc = preparar_oraculo_local(sub, sub.estado)
        desplaz = np.arange(N, dtype=np.int64)
        mec_mask = (1 << D) - 1

        def f_batch(masks):
            # Partir la máscara en Python ANTES de bajar a int64: con V = D + N
            # grande la máscara completa no entra en un C long, cada mitad sí.
            mec = np.fromiter(
                (int(m) & mec_mask for m in masks), dtype=np.int64, count=len(masks)
            )
            altos = np.fromiter(
                (int(m) >> D for m in masks), dtype=np.int64, count=len(masks)
            )
            return f_cara_batch_local(
                orc, mec, ((altos[:, None] >> desplaz) & 1).astype(bool)
            )

        _, mask, self._oraculo_stats = stoer_wagner_queyranne(
            D + N, f_batch, modo=self.modo
        )

        alcance_pos = {i for i in range(N) if (mask >> (D + i)) & 1}
        mec_pos = {j for j in range(D) if (mask >> j) & 1}

        # Reconstrucción con la marginal real, igual que hace `qsw` con `bipartir`
        # + `distribucion_marginal` — pero sobre las tablas locales, en O(2^kᵢ).
        dm = sub.bipartir_marginal(alcance_pos, mec_pos)
        perdida = emd_efecto(dm, dm_original)

        alcance = tuple(sub.indices[i] for i in sorted(alcance_pos))
        mecanismo = tuple(sub.dims[j] for j in sorted(mec_pos))

        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=float(perdida) if perdida != INFTY_POS else FLOAT_ZERO,
            distribucion_subsistema=dm_original,
            distribucion_particion=dm,
            particion=fmt_parts(
                (alcance, mecanismo), (sub.indices, sub.dims)
            ).strip(),
            tiempo_total=time.perf_counter() - t0,
            quiere_hablar=False,
        )
