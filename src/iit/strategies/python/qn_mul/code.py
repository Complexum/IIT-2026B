"""Estrategia QNodes-MUL: algoritmo Q con oráculo Zeta, paralelizado con multiprocessing.

Mismo algoritmo Q que ``qn`` (singletons + pares colgantes), incluyendo el **mismo
oráculo Zeta** (``qn.oracle.preparar_oraculo`` / ``f_cara``) para rankear candidatos
durante el MAO — bit-idéntico a ``qn``. La evaluación de cada **batch de cortes
independientes** (pre-pass de singletons y los ``f_cara(delta ∪ omega)`` de cada paso
MAO) se reparte entre procesos con ``multiprocessing.Pool``. El driver sigue secuencial
(orden MAO + contracciones dependen del paso previo); solo el cálculo de
``f_cara`` se paraleliza.

Al final, igual que ``qn``, se recalcula ``(distribucion_particion, perdida)`` del MIP
ganador con ``System.bipartir + distribucion_marginal + emd_efecto`` reales.
"""

import multiprocessing as mp
import os
import time

from src.iit.base.app import aplicacion

from src.iit.base.consts import ACTUAL, EFFECT, FLOAT_ZERO, INFTY_POS, INT_ZERO
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.qn.oracle import Oraculo, f_cara, preparar_oraculo
from src.iit.strategies.python.sia import SIA

Vertice = tuple[int, int]
Corte = tuple[tuple[int, ...], tuple[int, ...]]  # (alcance, mecanismo)

# Umbral: por debajo, evaluar inline (el IPC no compensa).
MIN_BATCH_PARALELO = 16

# ── Estado del worker (poblado por el initializer del Pool) ────────────────
_W_ORACULO: Oraculo | None = None


def _init_worker(oraculo: Oraculo) -> None:
    global _W_ORACULO
    _W_ORACULO = oraculo


def _eval_cut(corte: Corte):
    """Worker: EMD del corte (alcance, mecanismo) vía oráculo Zeta."""
    alcance, mecanismo = corte
    emd = f_cara(_W_ORACULO, alcance, mecanismo)
    return corte, emd


class QNodesMUL(SIA, nombre="qn_mul"):
    """MIP via algoritmo Q con evaluación de cortes en paralelo (multiprocessing)."""

    def resolver(self) -> Solution:
        t0 = time.perf_counter()
        dm_original = self.distribucion

        indices = self.sistema.indices
        dims = self.sistema.dims

        if not indices or not dims:
            return Solution(
                estrategia=self.nombre.capitalize(),
                perdida=FLOAT_ZERO,
                distribucion_subsistema=dm_original,
                distribucion_particion=dm_original,
                particion=fmt_parts(((), ()), ((), ())).strip(),
                tiempo_total=time.perf_counter() - t0,
                quiere_hablar=False,
            )

        # Precómputo Zeta: todas las sumas de cara en O(D·N·2^D), una sola vez.
        self._oraculo = preparar_oraculo(self.sistema)

        futuro = tuple((EFFECT, idx) for idx in indices)
        presente = tuple((ACTUAL, dim) for dim in dims)

        self.memoria_bipart: dict[Corte, float] = {}
        self.memoria_grupo_candidato: dict[Corte, float] = {}

        n_workers = max(1, os.cpu_count() or 1)
        vertices = list(presente + futuro)

        if n_workers <= 1:
            self._pool = None
            mip = self.algorithm(vertices)
        else:
            platform = "fork" if aplicacion.op_system == "macos" else "spawn"
            ctx = mp.get_context(platform)
            with ctx.Pool(
                processes=n_workers,
                initializer=_init_worker,
                initargs=(self._oraculo,),
            ) as pool:
                self._pool = pool
                mip = self.algorithm(vertices)

        # Reconstrucción exacta del MIP ganador: una marginalización real + EMD real.
        alcance, mecanismo = mip
        dist_mip = self.sistema.bipartir(alcance, mecanismo).distribucion_marginal()
        perdida_mip = float(emd_efecto(dist_mip, dm_original))
        texto = fmt_parts((alcance, mecanismo), (indices, dims))

        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=perdida_mip,
            distribucion_subsistema=dm_original,
            distribucion_particion=dist_mip,
            particion=texto.strip(),
            tiempo_total=time.perf_counter() - t0,
            quiere_hablar=False,
        )

    # ── Algoritmo Q ────────────────────────────────────────────────────────
    def algorithm(self, vertices: list):
        # Pre-pass: corte que aísla cada nodo individual (batch paralelo).
        self._registrar_batch([[v] for v in vertices])

        while len(vertices) > 1:
            omegas_ciclo: list = [vertices[0]]
            deltas_ciclo: list = vertices[1:]

            while len(deltas_ciclo) > 1:
                # Batch: por cada delta_k necesitamos loss(delta) y loss(delta ∪ omega).
                grupos = []
                for dk in deltas_ciclo:
                    grupos.append([dk])
                    grupos.append([dk, *omegas_ciclo])
                self._eval_cuts(grupos)

                emd_local = INFTY_POS
                indice_mip = INT_ZERO
                for k, dk in enumerate(deltas_ciclo):
                    emd_delta = self.memoria_bipart[self.__cut_key([dk])]
                    emd_union = self.memoria_bipart[
                        self.__cut_key([dk, *omegas_ciclo])
                    ]
                    ganancia = emd_union - emd_delta
                    if ganancia < emd_local:
                        emd_local = ganancia
                        indice_mip = k
                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            colgante = deltas_ciclo[INT_ZERO]
            self._registrar_batch([[colgante]])

            s_node = omegas_ciclo.pop()
            supernodo = self.__as_list(s_node) + self.__as_list(colgante)
            omegas_ciclo.append(supernodo)
            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda c: self.memoria_grupo_candidato[c],
        )

    # ── Evaluación de batches de cortes ────────────────────────────────────
    def _eval_cuts(self, grupos: list) -> None:
        """Asegura en ``memoria_bipart`` el EMD de cada grupo (dedup + cache + pool)."""
        pendientes = []
        vistos = set()
        for g in grupos:
            corte = self.__cut_key(g)
            if corte not in self.memoria_bipart and corte not in vistos:
                vistos.add(corte)
                pendientes.append(corte)

        if not pendientes:
            return

        if self._pool is None or len(pendientes) < MIN_BATCH_PARALELO:
            for corte in pendientes:
                alcance, mecanismo = corte
                self.memoria_bipart[corte] = f_cara(self._oraculo, alcance, mecanismo)
        else:
            for corte, emd in self._pool.map(_eval_cut, pendientes):
                self.memoria_bipart[corte] = emd

    def _registrar_batch(self, grupos: list) -> None:
        """Registra cada grupo como partición candidata (corte que lo aísla)."""
        self._eval_cuts(grupos)
        for g in grupos:
            corte = self.__cut_key(g)
            self.memoria_grupo_candidato[corte] = self.memoria_bipart[corte]

    # ── Claves y aplanado ──────────────────────────────────────────────────
    def __cut_key(self, grupo) -> Corte:
        """Corte canónico (alcance, mecanismo) que aísla la unión de ``grupo``."""
        alcance: list[int] = []
        mecanismo: list[int] = []
        for tiempo, valor in self.__flatten(grupo):
            (alcance if tiempo == EFFECT else mecanismo).append(valor)
        return tuple(sorted(alcance)), tuple(sorted(mecanismo))

    def __as_list(self, nodo) -> list:
        if isinstance(nodo, tuple) and len(nodo) == 2 and isinstance(nodo[0], int):
            return [nodo]
        return list(nodo)

    def __flatten(self, nodo):
        if isinstance(nodo, tuple) and len(nodo) == 2 and isinstance(nodo[0], int):
            yield nodo
        else:
            for sub in nodo:
                yield from self.__flatten(sub)
