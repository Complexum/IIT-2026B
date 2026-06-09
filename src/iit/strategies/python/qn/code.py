"""Estrategia QNodes (qn): MIP via algoritmo Q con oráculo Zeta de caras.

Algoritmo Q (estilo Queyranne) **exacto sobre EMD**, acelerado: en lugar de
remarginalizar (``bipartir + distribucion_marginal + emd_efecto``) en cada
llamada del oráculo, precomputa **una sola vez** todas las sumas de hiper-cara con
la transformada Zeta (igual que la estrategia ``analytic``) y sirve cada consulta
del oráculo en O(1) amortizado por cubo. Ver ``oracle.md`` para la derivación.

Identidad clave (repertorio de efecto, normalización firmada δ = H − p):

    EMD(alcance, mecanismo) = Σ_i  |mean_{F_i}(δ_i)|,
        F_i = complemento(mecanismo)  si  i ∈ alcance
        F_i = mecanismo               si  i ∉ alcance

donde mean_F(δ_i) = S_h(i, F) / 2^|F| sale de ``sumas[i, mask]`` precomputado.
Esto reproduce exactamente ``emd_efecto(bipartir(alc, mec).distribucion_marginal(),
distribucion)`` — verificado contra la ruta original (ver test de equivalencia).

Trabaja sobre vértices ``(tiempo, valor)`` de dos capas temporales:
  - ``(EFFECT, idx)`` con ``idx ∈ sistema.indices``  → candidatos a *alcance* (t+1).
  - ``(ACTUAL, dim)`` con ``dim ∈ sistema.dims``      → candidatos a *mecanismo* (t).

Candidatos evaluados:
  1. **Pre-pass de singletons**: el corte que aísla cada nodo individual.
  2. **Pares colgantes**: por cada fase, el MAO (omega crece eligiendo el delta de
     menor ganancia marginal) deja un nodo colgante cuyo corte real se evalúa; luego
     s y el colgante se contraen en un supernodo y el problema se reduce.

El MIP es la partición de menor EMD entre todos los candidatos. Es un subconjunto
de las 2^(m+n) biparticiones (heurística): exacta cuando el óptimo cae en ese
subconjunto — el pre-pass de singletons cubre la causa de fallo más común.
"""

import time
from typing import Union

import numpy as np

from src.iit.base.consts import ACTUAL, EFFECT, FLOAT_ZERO, INFTY_POS, INT_ZERO
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.strategies.python.analytic.code import hyperfaces
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.sia import SIA

# Un vértice es una 2-tupla (tiempo, valor); un grupo es una lista (anidable) de vértices.
Vertice = tuple[int, int]


class QNodes(SIA, nombre="qn"):
    """MIP via algoritmo Q con oráculo Zeta de caras (O(1) amortizado por consulta)."""

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
        self.__preparar_oraculo()

        futuro = tuple((EFFECT, idx) for idx in indices)
        presente = tuple((ACTUAL, dim) for dim in dims)

        self.clave_submodular: tuple[list[int], list[int]] = ([], [])
        self.memoria_bipart: dict = {}  # clave canónica → emd (float)
        self.memoria_grupo_candidato: dict = {}  # clave grupo → (emd, (alc, mec))

        vertices = list(presente + futuro)
        mip = self.algorithm(vertices)

        # Reconstrucción exacta del MIP ganador: una marginalización real + EMD real.
        _, (alcance, mecanismo) = self.memoria_grupo_candidato[mip]
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

    # ── Oráculo Zeta (caras precomputadas) ─────────────────────────────────
    def __preparar_oraculo(self) -> None:
        """Precomputa ``sumas[i, m]`` (Zeta sobre δ = H − p) y los mapas auxiliares."""
        sistema = self.sistema
        dims = sistema.dims
        self._D = len(dims)
        self._full_mask = (1 << self._D) - 1
        # mask bit d ↔ dims[d]  (mismo convenio que analytic.hyperfaces)
        self._pos_dim = {d: i for i, d in enumerate(dims)}
        self._indices_order = np.fromiter(
            (c.indice for c in sistema.ncubos), dtype=np.int64
        )

        data_nd = np.stack([c.ndata for c in sistema.ncubos])
        N = data_nd.shape[0]
        pivot_idx = tuple(int(sistema.estado_inicial[d]) for d in dims)
        pivot_vals = data_nd[(slice(None),) + pivot_idx]  # (N,)
        # Normalización firmada: δ = H − p (pivote queda en 0).
        delta_nd = data_nd - pivot_vals.reshape((N,) + (1,) * self._D)
        self._sumas = hyperfaces(N, self._D, delta_nd, pivot_idx)

    def __f_cara(
        self, alcance: tuple[int, ...], mecanismo: tuple[int, ...]
    ) -> float:
        """EMD del corte ``(alcance, mecanismo)`` leído de las sumas precomputadas.

        Reproduce ``emd_efecto(bipartir(alc, mec).distribucion_marginal(), ρ)``:
        cada cubo aporta |mean_{complemento(mec)}(δ)| si está en ``alcance``,
        o |mean_{mec}(δ)| en caso contrario.
        """
        m = 0
        for d in mecanismo:
            m |= 1 << self._pos_dim[d]
        cmask = self._full_mask ^ m
        sz_a = bin(m).count("1")

        val_a = np.abs(self._sumas[:, m]) / (1 << sz_a)            # |mean_mec(δ)|
        val_b = np.abs(self._sumas[:, cmask]) / (1 << (self._D - sz_a))  # |mean_compl(δ)|

        if alcance:
            in_alc = np.isin(
                self._indices_order, np.fromiter(alcance, dtype=np.int64)
            )
            cost = np.where(in_alc, val_b, val_a)
        else:
            cost = val_a
        return float(cost.sum())

    # ── Algoritmo Q ────────────────────────────────────────────────────────
    def algorithm(self, vertices: list):
        """Evalúa singletons + pares colgantes y retorna la clave (en
        ``memoria_grupo_candidato``) de menor EMD."""
        # Pre-pass: corte que aísla cada nodo individual.
        for v in vertices:
            self.__registrar_candidato(v)

        while len(vertices) > 1:
            omegas_ciclo: list = [vertices[0]]
            deltas_ciclo: list = vertices[1:]

            # Maximum Adjacency Ordering: omega absorbe el delta de menor ganancia.
            while len(deltas_ciclo) > 1:
                emd_local = INFTY_POS
                indice_mip = INT_ZERO
                for k in range(len(deltas_ciclo)):
                    emd_union, emd_delta = self.funcion_submodular(
                        deltas_ciclo[k], omegas_ciclo
                    )
                    ganancia = emd_union - emd_delta
                    if ganancia < emd_local:
                        emd_local = ganancia
                        indice_mip = k
                omegas_ciclo.append(deltas_ciclo[indice_mip])
                deltas_ciclo.pop(indice_mip)

            # Par colgante: el delta sobrante es el nodo colgante t.
            colgante = deltas_ciclo[INT_ZERO]
            self.__registrar_candidato(colgante)

            # Contraer s (último de omega) y t en un supernodo.
            s_node = omegas_ciclo.pop()
            supernodo = self.__as_list(s_node) + self.__as_list(colgante)
            omegas_ciclo.append(supernodo)
            vertices = omegas_ciclo

        return min(
            self.memoria_grupo_candidato,
            key=lambda clave: self.memoria_grupo_candidato[clave][INT_ZERO],
        )

    # ── EMD de biparticiones (vía oráculo Zeta) ────────────────────────────
    def funcion_submodular(
        self,
        delta: Union[Vertice, list],
        omegas: list,
    ) -> tuple[float, float]:
        """Retorna ``(emd_union, emd_delta)``: EMD de aislar ``delta ∪ omega`` y de
        aislar solo ``delta``. La ganancia ``emd_union - emd_delta`` guía el MAO."""
        emd_delta, _ = self.__emd_grupo([delta])
        emd_union, _ = self.__emd_grupo([delta, *omegas])
        return emd_union, emd_delta

    def __emd_grupo(self, grupos: list) -> tuple[float, tuple]:
        """EMD real del corte que aísla la unión de ``grupos`` (leída de ``sumas``).

        Retorna ``(emd, (alcance, mecanismo))``.
        """
        self.clave_submodular = ([], [])
        for g in grupos:
            self.definir_clave(g)
        alcance = tuple(self.clave_submodular[EFFECT])
        mecanismo = tuple(self.clave_submodular[ACTUAL])
        clave = (alcance, mecanismo)

        cacheado = self.memoria_bipart.get(clave)
        if cacheado is not None:
            return cacheado, clave

        emd = self.__f_cara(alcance, mecanismo)
        self.memoria_bipart[clave] = emd
        return emd, clave

    def __registrar_candidato(self, grupo) -> None:
        """Registra el corte que aísla ``grupo`` como partición candidata."""
        emd, clave = self.__emd_grupo([grupo])
        self.memoria_grupo_candidato[self.__clave_grupo(grupo)] = (emd, clave)

    # ── Claves y aplanado ──────────────────────────────────────────────────
    def definir_clave(self, conjunto: Union[Vertice, list]):
        """Acumula valores en ``clave_submodular[tiempo]`` (ordenado)."""
        for tiempo, valor in self.__flatten(conjunto):
            self.clave_submodular[tiempo].append(valor)
        self.clave_submodular[ACTUAL].sort()
        self.clave_submodular[EFFECT].sort()
        return self.clave_submodular

    def __clave_grupo(self, grupo) -> tuple:
        """Clave canónica hashable: vértices aplanados y ordenados."""
        return tuple(sorted(self.__flatten(grupo)))

    def __as_list(self, nodo) -> list:
        """Normaliza un vértice o grupo a lista de vértices/grupos."""
        if isinstance(nodo, tuple) and len(nodo) == 2 and isinstance(nodo[0], int):
            return [nodo]
        return list(nodo)

    def __flatten(self, nodo):
        """Aplana un vértice o grupo (anidado) a una secuencia de vértices."""
        if isinstance(nodo, tuple) and len(nodo) == 2 and isinstance(nodo[0], int):
            yield nodo
        else:
            for sub in nodo:
                yield from self.__flatten(sub)

    def __particion_a_grupos(self, mip) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Traduce la clave MIP (vértices) a ``(alcance, mecanismo)``."""
        alcance: list[int] = []
        mecanismo: list[int] = []
        for tiempo, valor in self.__flatten(mip):
            (alcance if tiempo == EFFECT else mecanismo).append(valor)
        return tuple(sorted(alcance)), tuple(sorted(mecanismo))
