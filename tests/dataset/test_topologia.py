"""Topología de la red (`src/io/topologia.py`).

Lo que se fija acá es la **separabilidad**: si existe un corte cerrado —un conjunto
de dims `A` donde cada nodo tiene todos o ninguno de sus padres— entonces `f = 0`
por construcción, sin importar los valores de las tablas. Una red así es trivial
como caso de prueba.
"""

import numpy as np
import pytest

from src.io.fst import generar_espec, generar_espec_pbn
from src.io.topologia import (
    MODOS,
    clausura,
    corte_cerrado,
    elegir_padres,
    es_separable,
)
from src.iit.strategies.python.zeta_local import (
    f_cara_batch_local,
    preparar_oraculo_local,
)


class TestEleccionDePadres:
    @pytest.mark.parametrize("modo", MODOS)
    @pytest.mark.parametrize("k", [2, 3])
    def test_fan_in_pedido(self, modo, k):
        padres = elegir_padres(12, k, np.random.default_rng(0), modo=modo)
        assert all(len(p) == k for p in padres)

    def test_anillo_tiene_columna_vertebral(self):
        """Cada nodo `i` toma `{i−1, i}` fijos: eso es lo que cierra el ciclo."""
        padres = elegir_padres(8, 2, np.random.default_rng(0), modo="anillo")
        for i, p in enumerate(padres):
            assert {(i - 1) % 8, i} <= set(p)

    def test_anillo_exige_k_al_menos_2(self):
        with pytest.raises(ValueError, match="columna vertebral"):
            elegir_padres(8, 1, np.random.default_rng(0), modo="anillo")

    def test_modo_desconocido_lanza(self):
        with pytest.raises(ValueError, match="modo desconocido"):
            elegir_padres(8, 2, np.random.default_rng(0), modo="hipercubo")


class TestSeparabilidad:
    def test_anillo_nunca_es_separable(self):
        """La clausura de cualquier dim recorre el ciclo entero."""
        for k in (2, 3, 5):
            for s in range(10):
                padres = elegir_padres(20, k, np.random.default_rng(s), modo="anillo")
                assert not es_separable(padres)

    def test_aleatoria_con_k_chico_casi_siempre_lo_es(self):
        """Es la razón por la que el régimen disperso daba φ = 0 siempre."""
        cuantas = sum(
            es_separable(elegir_padres(20, 2, np.random.default_rng(s), modo="aleatoria"))
            for s in range(10)
        )
        assert cuantas == 10

    def test_clausura_crece_hasta_cerrar(self):
        # dos triángulos disjuntos: {0,1,2} y {3,4,5}
        padres = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
        assert clausura(padres, {0}) == {0, 1, 2}
        assert clausura(padres, {3}) == {3, 4, 5}

    def test_corte_cerrado_devuelve_a_y_alcance(self):
        padres = ((0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3))
        A, alcance = corte_cerrado(padres)
        assert A == frozenset({0, 1, 2})
        assert alcance == frozenset({0, 1, 2})  # los nodos con TODOS sus padres en A

    def test_grafo_conectado_no_tiene_corte_propio(self):
        padres = tuple(((i - 1) % 6, i) for i in range(6))
        assert corte_cerrado(padres) is None


class TestElCorteCerradoCuestaCero:
    """La razón de ser del módulo: un corte cerrado da f = 0 exacto, y eso no
    depende de los valores de la tabla."""

    @pytest.mark.parametrize(
        "hacer",
        [
            lambda: generar_espec(14, 2, topologia="aleatoria", semilla=5),
            lambda: generar_espec_pbn(14, 2, funciones=3, topologia="aleatoria", semilla=5),
            lambda: generar_espec_pbn(14, 2, ruido=0.2, topologia="aleatoria", semilla=5),
        ],
        ids=["binario", "pbn", "ruido"],
    )
    def test_f_es_cero_sobre_el_corte_cerrado(self, hacer):
        espec = hacer()
        hallado = corte_cerrado(espec.padres)
        assert hallado is not None, "la red de prueba debería ser separable"
        A, alcance = hallado

        orc = preparar_oraculo_local(espec, tuple(0 for _ in range(espec.n)))
        mec = np.array([sum(1 << d for d in A)], dtype=np.int64)
        alc = np.array([[i in alcance for i in range(espec.n)]], dtype=bool)
        assert f_cara_batch_local(orc, mec, alc)[0] == pytest.approx(0.0, abs=1e-12)
