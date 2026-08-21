"""Núcleo QSW: Stoer-Wagner × Queyranne sobre una función simétrica arbitraria.

Sólo el algoritmo: sin dependencias de SIA/System y sin nada de backends (eso vive
en ``backend.py``). Recibe ``V`` y un oráculo batch ``f_batch`` y devuelve
``(mejor_valor, mejor_mascara)``. Testeable en aislamiento (ver
``tests/strategies/test_qsw.py``, que lo alimenta con la función de corte de un
grafo real para validar el puente matemático).

Puente SW ↔ Queyranne
---------------------
Para una ``f`` de corte de grafo, ``f(A∪v) − f(v) = f(A) − 2·w(A,v)``. Como
``f(A)`` es constante durante el argmin sobre ``v``, **minimizar la ganancia de
Queyranne ≡ maximizar la adyacencia de Stoer-Wagner**. Son el mismo algoritmo.

Eso permite reemplazar las O(V³) llamadas al oráculo de Queyranne por:

  1. un *seed* de O(V²) evaluaciones que estima la matriz de pesos
     ``W[u][v] = (f({u}) + f({v}) − f({u,v})) / 2``  (exacta si f es gráfica), y
  2. el MAO de Stoer-Wagner, cuyo update de key es ``key[w] += W[t][w]`` — O(1),
     sin oráculo.

El surrogate ``W`` sólo **guía** la búsqueda: toda candidata se re-puntúa con la
``f`` exacta antes de decidir, así que su error no contamina el resultado.
"""

from typing import Callable, Sequence

import numpy as np

NEG_INF = float("-inf")

# f_batch(masks) -> array (len(masks),) con f evaluada en cada máscara de vértices.
FBatch = Callable[[Sequence[int]], np.ndarray]


class OraculoCache:
    """Envuelve ``f_batch`` deduplicando máscaras y cacheando resultados.

    ``llamadas`` cuenta evaluaciones reales del oráculo (métrica del paper:
    O(V²) aquí vs O(V³) en Queyranne); ``batches`` cuenta llamadas a numpy.
    """

    __slots__ = ("_f", "_cache", "llamadas", "batches")

    def __init__(self, f_batch: FBatch) -> None:
        self._f = f_batch
        self._cache: dict[int, float] = {0: 0.0}
        self.llamadas = 0
        self.batches = 0

    def __call__(self, masks: Sequence[int]) -> np.ndarray:
        pendientes = [m for m in dict.fromkeys(masks) if m not in self._cache]
        if pendientes:
            vals = self._f(pendientes)
            for m, v in zip(pendientes, vals):
                self._cache[m] = float(v)
            self.llamadas += len(pendientes)
            self.batches += 1
        cache = self._cache
        return np.fromiter((cache[m] for m in masks), dtype=np.float64, count=len(masks))

    def uno(self, mask: int) -> float:
        return float(self(( mask,))[0])


def seed_weights(V: int, f: OraculoCache) -> tuple[np.ndarray, np.ndarray]:
    """Matriz de pesos de segundo orden + valores singleton. **1 batch numpy**.

    ``W[u][v] = (f({u}) + f({v}) − f({u,v})) / 2`` — exacta si ``f`` es gráfica.
    Puede tener entradas negativas cuando no lo es; el max-adjacency sigue
    funcionando (sólo ordena).
    """
    bits = [1 << u for u in range(V)]
    iu, ju = np.triu_indices(V, k=1)
    pares = [bits[u] | bits[v] for u, v in zip(iu, ju)]

    vals = f(bits + pares)
    singles = vals[:V]
    par_vals = vals[V:]

    W = np.zeros((V, V), dtype=np.float64)
    w_uv = (singles[iu] + singles[ju] - par_vals) / 2.0
    W[iu, ju] = w_uv
    W[ju, iu] = w_uv
    return W, singles


def mao_phase(W: np.ndarray) -> list[int]:
    """Maximum Adjacency Ordering con keys incrementales (Stoer-Wagner).

    Cada paso es un ``+=`` de vector y un ``argmax`` → O(n) numpy por paso,
    O(n²) por fase, y **cero llamadas al oráculo**. Es exactamente lo que
    Queyranne paga con O(n²) evaluaciones de ``f`` por fase.
    """
    n = W.shape[0]
    key = W[0].copy()
    key[0] = NEG_INF
    orden = [0]
    for _ in range(n - 1):
        u = int(np.argmax(key))
        orden.append(u)
        # -inf + finito = -inf, así que los ya elegidos nunca resucitan.
        key += W[u]
        key[u] = NEG_INF
    return orden


def stoer_wagner_queyranne(
    V: int,
    f_batch: FBatch,
    modo: str = "exacto",
    rondas_reparacion: int = 2,
) -> tuple[float, int, OraculoCache]:
    """Minimiza una función simétrica ``f: 2^V → R`` con ``f(∅) = f(V) = 0``.

    Args:
        V: número de vértices (máscaras de V bits sobre enteros de Python).
        f_batch: oráculo batch.
        modo: ``"exacto"`` recalcula con el oráculo la fila del supernodo tras
            cada contracción (sin drift, O(V²) llamadas en V batches);
            ``"estatico"`` usa el ``W[s] += W[t]`` de Stoer-Wagner puro (cero
            llamadas tras el seed → un único batch, 100% paralelizable).
        rondas_reparacion: rondas de 1-opt (Kernighan-Lin) sobre la ganadora.

    Returns:
        ``(mejor_valor, mejor_mascara, oraculo)``. ``mejor_mascara`` nunca es
        ∅ ni el conjunto completo (ambos triviales con f = 0).
    """
    if modo not in ("exacto", "estatico"):
        raise ValueError(f"modo desconocido: {modo!r} (usar 'exacto' | 'estatico')")

    f = OraculoCache(f_batch)
    full = (1 << V) - 1
    if V < 2:
        return 0.0, 0, f

    W, singles = seed_weights(V, f)

    # Pre-pass de singletons. Cubre el baseline C_conc de `analytic` sin caso
    # especial: alcance={un nodo}, mecanismo=∅ ES el singleton (EFFECT, i).
    candidatas: list[int] = [1 << u for u in range(V)]

    miembros = [1 << u for u in range(V)]
    f_super = singles.astype(np.float64).copy()

    while len(miembros) > 1:
        orden = mao_phase(W)
        s, t = orden[-2], orden[-1]

        # Par colgante: el corte que aísla t es candidata de Queyranne.
        candidatas.append(miembros[t])

        # ── Contracción s ← s ∪ t ───────────────────────────────────────────
        miembros[s] |= miembros[t]
        n = len(miembros)

        if modo == "estatico":
            W[s, :] += W[t, :]
            W[:, s] += W[:, t]
        else:
            m_s = miembros[s]
            otros = [v for v in range(n) if v != s and v != t]
            f_s = f((m_s,))[0]
            f_super[s] = f_s
            if otros:
                unions = f([m_s | miembros[v] for v in otros])
                fila = (f_s + f_super[otros] - unions) / 2.0
                W[s, otros] = fila
                W[otros, s] = fila
        W[s, s] = 0.0

        keep = [v for v in range(n) if v != t]
        W = W[np.ix_(keep, keep)]
        f_super = f_super[keep]
        miembros = [miembros[v] for v in keep]

    mejor_val, mejor_mask = puntuar_candidatas(V, f, candidatas, rondas_reparacion)
    return mejor_val, mejor_mask, f


def puntuar_candidatas(
    V: int,
    f: OraculoCache,
    candidatas: list[int],
    rondas_reparacion: int = 2,
) -> tuple[float, int]:
    """Re-scoring exacto + reparación 1-opt. El surrogate guió, la ``f`` real decide.

    Compartido por la ruta Python y la ruta C: el kernel C sólo **busca** y devuelve
    candidatas; la decisión final siempre se toma acá con la ``f`` exacta.
    """
    full = (1 << V) - 1

    def evaluar(masks: list[int]) -> tuple[float, int]:
        validas = [m for m in dict.fromkeys(masks) if m != 0 and m != full]
        if not validas:
            return float("inf"), 0
        vals = f(validas)
        k = int(np.argmin(vals))
        return float(vals[k]), validas[k]

    mejor_val, mejor_mask = evaluar(candidatas)

    # Reparación 1-opt: mover un vértice de lado a la vez (Kernighan-Lin).
    for _ in range(rondas_reparacion):
        vecinos = [mejor_mask ^ (1 << u) for u in range(V)]
        val, mask = evaluar(vecinos)
        if val >= mejor_val:
            break
        mejor_val, mejor_mask = val, mask

    return mejor_val, mejor_mask
