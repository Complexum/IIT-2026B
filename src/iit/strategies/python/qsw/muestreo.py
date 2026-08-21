"""Oráculo por muestreo: medias de cara sin materializar las 2^D caras.

Única responsabilidad de este módulo. El algoritmo (`core.py`), la selección de
backend (`backend.py`) y el glue con SIA (`code.py`) no se tocan.

Por qué
-------
La búsqueda de QSW es polinómica, pero el **oráculo** no: construir la tabla de
caras con la transformada Zeta cuesta `Θ(D·N·2^D)` y, medido, es el 100 % del
tiempo de `qsw` (a n=22, 0.957 s de 0.946 s totales). Por eso `qsw` sólo le gana
~1.3× a `analytic`: la búsqueda deja de importar cuando el precómputo manda.

Que "Queyranne es O(D³)" se refiere a **llamadas al oráculo**, no a tiempo. Y hay
un piso duro de `Ω(N·2^D)` porque la entrada misma son N·2^D probabilidades.
La única salida es **no calcular todas las caras**.

Qué hace
--------
`f_cara` sólo necesita `mean_A(δ_i)`: el promedio de δ_i sobre la sub-cara donde
las dims de A quedan libres y el resto fija en el pivote. En vez de sumar las
2^|A| celdas, se promedian K muestreadas — estimador insesgado con desviación
`σ_i/√K`, independiente de 2^|A|.

Lo que lo vuelve barato de verdad: **las máscaras de cara distintas dependen sólo
de la parte mecanismo** del corte (el alcance únicamente elige cuál de las dos
medias usa cada nodo). El seed de QSW pide singletons y pares, así que hay
`O(D²)` máscaras distintas, no `O(V²)`. Costo total `O(N·K·D²)`.

Medido en el prototipo, contra `analytic` y con φ idéntico: 6.7× a n=18 y 23.5×
a n=20 — el speedup crece exponencialmente con n, que es exactamente lo que
faltaba.

Exactitud
---------
Aproximado por construcción, con tres guardas para que **no se degrade cuando la
red crece**:

1. `K` escala con el tamaño: `max(K_base, c·N)`. Fijarlo chico falla — a n=16 con
   K=256 el prototipo erró un 68 %.
2. Las caras chicas (`2^|A| <= K`) se suman **exactas**: muestrear más puntos que
   celdas tiene no tiene sentido, y ahí es donde vive el error del caso anterior.
3. Las candidatas finales se re-puntúan con `8·K`, que es donde se decide.

Todo sigue siendo polinómico: `O(N²·D²)`.

Además se reporta el ruido: la desviación estimada del margen entre la ganadora y
la segunda. Si el margen no supera el ruido, se avisa en vez de fingir exactitud.
"""

import numpy as np

# K nunca baja de acá, y crece con N para que la precisión no se degrade al
# crecer la red. Calibrados con `benchmarks/calibrar_k.py`.
K_BASE = 2048
K_POR_NODO = 128
FACTOR_RESCORING = 8


def k_efectivo(N: int, k: str | int = "auto") -> int:
    """`auto` → `max(K_BASE, K_POR_NODO·N)`; un entero se respeta tal cual."""
    if k == "auto" or k is None:
        return max(K_BASE, K_POR_NODO * N)
    return int(k)


class OraculoMuestreado:
    """Estima `f(S)` muestreando cada cara en vez de sumarla entera.

    Expone la misma interfaz batch que espera `core.stoer_wagner_queyranne`:
    `f_batch(masks) -> np.ndarray`, con máscaras de V = D+N bits
    (bits `0..D-1` = mecanismo, bits `D..D+N-1` = alcance).
    """

    def __init__(self, sistema, k: str | int = "auto", seed: int = 0) -> None:
        self.D = D = len(sistema.dims)
        self.N = N = len(sistema.ncubos)
        self.mec_mask = (1 << D) - 1
        self.K = k_efectivo(N, k)
        self._rng = np.random.default_rng(seed)

        flat = np.stack([c.data for c in sistema.ncubos])
        self._flat = flat.astype(np.float32, copy=flat.dtype != np.float32)
        self._pivot_flat = sum(
            (int(sistema.estado_inicial[d]) & 1) << j for j, d in enumerate(sistema.dims)
        )
        self._piv = self._flat[:, self._pivot_flat][:, None]

        self._cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
        self._desplaz = np.arange(N, dtype=np.int64)
        self.caras_exactas = 0
        self.caras_muestreadas = 0

    # ── medias de cara ──────────────────────────────────────────────────────
    def _mean_cara(self, mask: int, K: int) -> tuple[np.ndarray, np.ndarray]:
        """`(media, desviación_de_la_media)` de δ sobre la cara `mask`, por nodo.

        Devuelve desviación 0 cuando la cara se sumó exacta.
        """
        clave = (mask, K)
        hit = self._cache.get(clave)
        if hit is not None:
            return hit

        pc = int(mask).bit_count()
        fijos = self._pivot_flat & ~mask
        if (1 << pc) <= K:
            # Cara chica: enumerarla entera sale más barato que muestrearla, y es
            # exacta. Acá vive el fallo típico del muestreo con K bajo.
            libres = [j for j in range(self.D) if (mask >> j) & 1]
            u = np.arange(1 << pc, dtype=np.int64)
            despl = np.zeros(1 << pc, dtype=np.int64)
            for k, j in enumerate(libres):
                despl |= ((u >> k) & 1) << j
            pos = fijos | despl
            vals = self._flat[:, pos] - self._piv
            self.caras_exactas += 1
            res = (vals.mean(axis=1), np.zeros(self.N, dtype=np.float64))
        else:
            bruto = self._rng.integers(0, 1 << 62, size=K, dtype=np.int64)
            pos = fijos | (bruto & mask)
            vals = self._flat[:, pos] - self._piv
            self.caras_muestreadas += 1
            # error estándar de la media: σ/√K
            res = (vals.mean(axis=1), vals.std(axis=1) / np.sqrt(K))

        self._cache[clave] = res
        return res

    # ── interfaz batch ──────────────────────────────────────────────────────
    def _evaluar(self, masks, K: int, con_ruido: bool):
        out = np.empty(len(masks), dtype=np.float64)
        ruido = np.empty(len(masks), dtype=np.float64) if con_ruido else None
        for k, m in enumerate(masks):
            m = int(m)
            mec = m & self.mec_mask
            mean_a, sd_a = self._mean_cara(mec, K)
            mean_b, sd_b = self._mean_cara(self.mec_mask ^ mec, K)
            en_alcance = (((m >> self.D) >> self._desplaz) & 1).astype(bool)
            out[k] = np.where(en_alcance, np.abs(mean_b), np.abs(mean_a)).sum()
            if con_ruido:
                # |·| no cambia la varianza de primer orden; los nodos se suman
                # como independientes.
                v = np.where(en_alcance, sd_b, sd_a)
                ruido[k] = float(np.sqrt((v**2).sum()))
        return (out, ruido) if con_ruido else out

    def f_batch(self, masks) -> np.ndarray:
        return self._evaluar(masks, self.K, con_ruido=False)

    def repuntuar(self, masks) -> tuple[np.ndarray, np.ndarray]:
        """Re-scoring de las candidatas finales con `FACTOR_RESCORING·K` muestras."""
        return self._evaluar(masks, self.K * FACTOR_RESCORING, con_ruido=True)

    # ── diagnóstico ─────────────────────────────────────────────────────────
    def confianza(self, masks) -> dict:
        """¿El margen entre la ganadora y la segunda supera el ruido del muestreo?

        Devuelve el veredicto en vez de afirmar exactitud: con muestreo el
        resultado es una estimación, y el usuario merece saber cuándo es dudosa.
        """
        validas = [m for m in dict.fromkeys(int(x) for x in masks) if m]
        vals, ruido = self.repuntuar(validas)
        orden = np.argsort(vals)
        mejor = int(orden[0])
        if len(orden) < 2:
            return {"mask": validas[mejor], "valor": float(vals[mejor]),
                    "margen": float("inf"), "ruido": 0.0, "confiable": True}
        segundo = int(orden[1])
        margen = float(vals[segundo] - vals[mejor])
        sigma = float(np.sqrt(ruido[mejor] ** 2 + ruido[segundo] ** 2))
        return {
            "mask": validas[mejor],
            "valor": float(vals[mejor]),
            "margen": margen,
            "ruido": sigma,
            # 2σ ≈ 95 %: por debajo de eso la ganadora podría ser la segunda.
            "confiable": margen > 2.0 * sigma,
        }
