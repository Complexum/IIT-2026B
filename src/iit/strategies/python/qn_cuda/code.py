"""Estrategia QNodes-CUDA: algoritmo Q con oráculo Zeta, biparticiones batched en GPU (cupy).

Mismo algoritmo Q que ``qn`` (singletons + pares colgantes), incluyendo el **mismo
oráculo Zeta** (``qn.oracle.preparar_oraculo`` / ``sumas[i, m] = Σ`` sobre δ = H − p) para
rankear candidatos durante el MAO. El precompute corre en CPU (idéntico a ``qn``,
``qn_mul`` y ``qn_mpi``); ``sumas`` se sube una vez a GPU y cada **batch de cortes
independientes** se evalúa de una sola pasada vectorizada.

Reducción exacta del corte
--------------------------
``f_cara(alcance, mecanismo) = Σ_i |val_b_i|`` si ``i ∈ alcance`` (media de
``sumas[i, ~m]``, complemento de ``mecanismo``) o ``Σ_i |val_a_i|`` si ``i ∉ alcance``
(media de ``sumas[i, m]``) — misma fórmula que ``qn.oracle.f_cara``, evaluada en lote
con ``cp.where``.

Al final, igual que ``qn``, se recalcula ``(distribucion_particion, perdida)`` del MIP
ganador con ``System.bipartir + distribucion_marginal + emd_efecto`` reales (CPU,
costo O(N·2^D) una sola vez).

Hard-fail: sin ``cupy``/GPU, ``resolver`` lanza ``RuntimeError`` (la estrategia sigue
registrada y visible en CLI). Instalación cluster: ``uv pip install -e ".[cuda]"``.
"""

import time

import numpy as np

from src.iit.base.consts import ACTUAL, EFFECT, FLOAT_ZERO, INFTY_POS, INT_ZERO
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.qn.oracle import preparar_oraculo
from src.iit.strategies.python.sia import SIA

Vertice = tuple[int, int]
Corte = tuple[tuple[int, ...], tuple[int, ...]]  # (alcance, mecanismo)

D_MAX_CUDA = 27  # sumas (N, 2^D) float32; cota de memoria GPU


class QNodesCUDA(SIA, nombre="qn_cuda"):
    """MIP via algoritmo Q con biparticiones evaluadas en lote sobre GPU (cupy)."""

    def resolver(self) -> Solution:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("sin dispositivos CUDA")
        except Exception as e:
            raise RuntimeError(
                "qn_cuda requiere CUDA + cupy. Instala '.[cuda]' en un nodo con GPU."
            ) from e

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

        D = len(dims)
        if D > D_MAX_CUDA:
            raise RuntimeError(
                f"qn_cuda: D={D} excede D_MAX_CUDA={D_MAX_CUDA} (memoria GPU)."
            )

        # ── Precompute Zeta en CPU (idéntico a qn), sumas subidas a GPU ──────
        oraculo = preparar_oraculo(self.sistema)
        self._cp = cp
        self._D = oraculo.D
        self._full = oraculo.full_mask
        self._pos_dim = oraculo.pos_dim
        self._indices_order = oraculo.indices_order
        self._sumas_g = cp.asarray(oraculo.sumas)
        self._pop = cp.asarray(
            np.array([bin(m).count("1") for m in range(1 << D)], dtype=np.float32)
        )

        self.memoria_bipart: dict[Corte, float] = {}
        self.memoria_grupo_candidato: dict[Corte, float] = {}

        futuro = tuple((EFFECT, idx) for idx in indices)
        presente = tuple((ACTUAL, dim) for dim in dims)
        vertices = list(presente + futuro)
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
        self._registrar_batch([[v] for v in vertices])

        while len(vertices) > 1:
            omegas_ciclo: list = [vertices[0]]
            deltas_ciclo: list = vertices[1:]

            while len(deltas_ciclo) > 1:
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

    # ── Evaluación batched en GPU (oráculo Zeta, igual a qn.oracle.f_cara) ──
    def _eval_cuts(self, grupos: list) -> None:
        cp = self._cp
        pendientes: list[Corte] = []
        vistos = set()
        for g in grupos:
            corte = self.__cut_key(g)
            if corte not in self.memoria_bipart and corte not in vistos:
                vistos.add(corte)
                pendientes.append(corte)
        if not pendientes:
            return

        B = len(pendientes)
        full = self._full
        masks = np.empty(B, dtype=np.int64)
        comps = np.empty(B, dtype=np.int64)
        alc_bool = np.zeros((len(self._indices_order), B), dtype=bool)
        for b, (alcance, mecanismo) in enumerate(pendientes):
            m = 0
            for d in mecanismo:
                m |= 1 << self._pos_dim[d]
            masks[b] = m
            comps[b] = full ^ m
            if alcance:
                alc_set = set(alcance)
                alc_bool[:, b] = np.isin(self._indices_order, list(alc_set))

        masks_g = cp.asarray(masks)
        comps_g = cp.asarray(comps)
        alc_g = cp.asarray(alc_bool)

        # |mean_mec(δ)| (val_a) y |mean_compl(δ)| (val_b) — igual a qn.oracle.f_cara
        val_a = cp.abs(self._sumas_g[:, masks_g]) / cp.exp2(self._pop[masks_g])[None, :]
        val_b = cp.abs(self._sumas_g[:, comps_g]) / cp.exp2(self._pop[comps_g])[None, :]
        cost = cp.where(alc_g, val_b, val_a)  # (N, B)
        loss = cost.sum(axis=0)  # (B,)

        loss_h = cp.asnumpy(loss)
        for b, corte in enumerate(pendientes):
            self.memoria_bipart[corte] = float(loss_h[b])

    def _registrar_batch(self, grupos: list) -> None:
        self._eval_cuts(grupos)
        for g in grupos:
            corte = self.__cut_key(g)
            self.memoria_grupo_candidato[corte] = self.memoria_bipart[corte]

    # ── Claves y aplanado ──────────────────────────────────────────────────
    def __cut_key(self, grupo) -> Corte:
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
