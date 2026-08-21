"""Oráculo por Zeta local (`zeta_local.py`): exacto, sin materializar `2^D`.

La propiedad que se fija acá es la igualdad **exacta** con el oráculo actual, no
una aproximación: si difieren, el atajo no sirve.
"""

import numpy as np
import pytest

from src.iit.core.params import Params
from src.iit.strategies.python.qn.oracle import f_cara_batch, preparar_oraculo
from src.iit.strategies.python.zeta_local import (
    f_cara_batch_local,
    preparar_oraculo_local,
)
from src.io.fst import espec_desde_cnf, generar_espec, tpm_nodo_estado
from src.io.manager import reducir_a_subsistema

# A=B∨C, B=A∧C, C=A xor B
RED_MANO = {
    1: [[2, 3]],
    2: [[1], [3]],
    3: [[1, 2], [-1, -2]],
}


def _todos_los_cortes(espec, estado):
    """Ambos oráculos evaluados en las 2^V máscaras. Devuelve `(ref, local)`."""
    n = espec.n
    tpm = tpm_nodo_estado(espec)
    params = Params("".join(str(b) for b in estado), "1" * n, "1" * n, "1" * n)
    sistema = reducir_a_subsistema(tpm, params)
    D, N = len(sistema.dims), len(sistema.indices)

    todas = np.arange(1 << (D + N), dtype=np.int64)
    mec = todas & ((1 << D) - 1)
    alc = ((todas[:, None] >> (D + np.arange(N))) & 1).astype(bool)

    ref = f_cara_batch(preparar_oraculo(sistema), mec, alc).astype(np.float64)
    loc = f_cara_batch_local(preparar_oraculo_local(espec, estado), mec, alc)
    return ref, loc


class TestEquivalenciaExacta:
    def test_red_escrita_a_mano(self):
        espec = espec_desde_cnf(RED_MANO)
        for s in range(1 << espec.n):
            estado = tuple((s >> j) & 1 for j in range(espec.n))
            ref, loc = _todos_los_cortes(espec, estado)
            np.testing.assert_array_equal(loc, ref)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_generada_n3(self, k):
        espec = generar_espec(3, k, semilla=7)
        for s in range(1 << 3):
            estado = tuple((s >> j) & 1 for j in range(3))
            ref, loc = _todos_los_cortes(espec, estado)
            np.testing.assert_array_equal(loc, ref)

    @pytest.mark.parametrize("k", [1, 2, 4])
    def test_generada_n5(self, k):
        espec = generar_espec(5, k, semilla=11)
        estado = (1, 0, 1, 1, 0)
        ref, loc = _todos_los_cortes(espec, estado)
        np.testing.assert_array_equal(loc, ref)


class TestNoMutaLaEspecificacion:
    def test_preparar_no_toca_las_tablas(self):
        """`ascontiguousarray` no copia si el array ya es contiguo y del dtype
        pedido: sin copia explícita, el `δ = H − p` y la Zeta mutaban la spec en
        sitio y cada llamada la corrompía un poco más."""
        espec = generar_espec(4, 2, semilla=3)
        antes = [t.copy() for t in espec.tablas]
        for _ in range(3):
            preparar_oraculo_local(espec, (0, 1, 0, 1))
        for a, b in zip(antes, espec.tablas):
            np.testing.assert_array_equal(a, b)

    def test_repetir_da_el_mismo_resultado(self):
        espec = generar_espec(4, 2, semilla=3)
        estado = (0, 1, 0, 1)
        primero = preparar_oraculo_local(espec, estado)
        tercero = [preparar_oraculo_local(espec, estado) for _ in range(3)][-1]
        for a, b in zip(primero.sumas, tercero.sumas):
            np.testing.assert_array_equal(a, b)


class TestCoste:
    def test_precomputo_escala_con_el_fan_in_no_con_n(self):
        chico = preparar_oraculo_local(generar_espec(30, 2, semilla=5), (0,) * 30)
        assert chico.coste_precomputo == 30 * 2 * 4  # Σ kᵢ·2^kᵢ
        # el materializado seria D·N·2^D = 30·30·2^30
        assert chico.coste_precomputo < 30 * 30 * (1 << 30) / 10**6

    def test_nodo_constante_no_aporta(self):
        """Sin padres, δ ≡ 0: el término es 0 y no rompe el índice de la tabla."""
        from src.io.fst import EspecFST

        espec = EspecFST(n=1, padres=((),), tablas=(np.array([1.0]),))
        orc = preparar_oraculo_local(espec, (0,))
        got = f_cara_batch_local(orc, np.array([0, 1]), np.array([[False], [True]]))
        np.testing.assert_array_equal(got, [0.0, 0.0])
