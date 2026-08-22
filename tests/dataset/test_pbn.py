"""Redes booleanas probabilísticas (`src/io/pbn.py`).

Modelo de Shmulevich et al. (2002): cada nodo elige al azar entre varias fórmulas,
así que `P(nodo=1|padres) = Σ c_l · f_l(padres)`. Lo que se fija acá es la regla de
discriminación del formato (el único punto ambiguo) y la compilación de la mezcla.
"""

import numpy as np
import pytest

from src.io import cnf, pbn
from src.io.fst import (
    cnf_de_espec,
    # elegir_padres,
    espec_desde_cnf,
    generar_espec_pbn,
    tpm_nodo_estado,
)

N = 3
# Nodo 1 determinista (B∨C); nodo 2 PBN; nodo 3 determinista (A xor B).
RED_MIXTA = {
    1: [[2, 3]],
    2: [[0.7, [[1], [3]]], [0.3, [[1, 2]]]],
    3: [[1, 2], [-1, -2]],
}


class TestDiscriminacion:
    """Una cláusula es una lista plana de enteros; una entrada PBN es
    `[probabilidad, lista_de_cláusulas]`. Es el único punto ambiguo del formato."""

    @pytest.mark.parametrize(
        "spec, esperado",
        [
            ([[2, 3]], False),                          # una cláusula de dos literales
            ([[1], [3]], False),                        # dos cláusulas unitarias
            ([[1, 2], [-1, -2]], False),                # xor
            ([], False),                                # constante 1
            ([[]], False),                              # constante 0
            ([[0.7, [[1]]], [0.3, [[2]]]], True),       # PBN
            ([[1.0, [[1], [3]]]], True),                # PBN de una sola función
        ],
    )
    def test_es_pbn(self, spec, esperado):
        assert pbn.es_pbn(spec) is esperado


class TestCompilacion:
    def test_mezcla_convexa(self):
        """tabla[m] = Σ c_l · f_l(m), sobre la UNIÓN de padres."""
        padres, tabla = pbn.compilar(RED_MIXTA[2], N, 2)
        assert padres == (0, 1, 2)  # unión de {0,2} (A∧C) y {0,1} (A∨B)
        # bit0=A, bit1=B, bit2=C. m=5 → A=1,B=0,C=1 → A∧C=1 y A∨B=1 → 0.7+0.3
        np.testing.assert_allclose(tabla[0b101], 1.0)
        np.testing.assert_allclose(tabla[0b001], 0.3)  # sólo A∨B
        np.testing.assert_allclose(tabla[0b000], 0.0)  # ninguna

    def test_probabilidades_deben_sumar_uno(self):
        with pytest.raises(ValueError, match="deben sumar 1"):
            pbn.compilar([[0.7, [[1]]], [0.2, [[2]]]], N, 1)

    def test_probabilidad_negativa_lanza(self):
        with pytest.raises(ValueError, match="negativa"):
            pbn.compilar([[1.3, [[1]]], [-0.3, [[2]]]], N, 1)

    def test_determinista_es_el_caso_de_una_funcion(self):
        _, mezcla = pbn.compilar([[1.0, [[2, 3]]]], N, 1)
        directa = cnf.evaluar([[2, 3]], cnf.variables_de([[2, 3]]))
        np.testing.assert_array_equal(mezcla, directa)


class TestGeneracion:
    @pytest.mark.parametrize("kw", [{"funciones": 2}, {"funciones": 3}, {"ruido": 0.1}])
    def test_fan_in_estricto_sobre_la_mezcla(self, kw):
        """Sin esto una tabla constante deja el nodo en fan-in 0: su CNF canónica es
        `[]` o `[[]]` y no menciona variables."""
        espec = generar_espec_pbn(8, 3, semilla=7, **kw)
        assert espec.fan_in == (3,) * 8

    @pytest.mark.parametrize("kw", [{"funciones": 3}, {"ruido": 0.2}])
    def test_produce_tablas_no_binarias(self, kw):
        assert not generar_espec_pbn(6, 2, semilla=7, **kw).es_binaria()

    def test_ruido_da_epsilon_y_uno_menos_epsilon(self):
        """P = ε + (1−2ε)·f, así que la tabla sólo toma esos dos valores."""
        espec = generar_espec_pbn(5, 2, ruido=0.1, semilla=7)
        for tabla in espec.tablas:
            assert set(np.round(tabla, 9).tolist()) <= {0.1, 0.9}

    def test_ruido_fuera_de_rango_lanza(self):
        with pytest.raises(ValueError, match="ruido debe estar"):
            generar_espec_pbn(4, 2, ruido=0.8, semilla=1)

    def test_topologia_es_la_que_se_le_pasa(self):
        """`generar_formulas` recibe los padres en vez de elegirlos: por eso no hay
        ciclo de importación con `fst` ni topología duplicada."""
        rng = np.random.default_rng(0)
        padres = ((0, 1), (1, 2), (0, 2))
        formulas = pbn.generar_formulas(padres, rng, funciones=2)
        espec = espec_desde_cnf(formulas)
        assert espec.padres == padres


class TestConvivencia:
    """Nodos deterministas y PBN en la misma red y el mismo archivo."""

    def test_red_mixta_compila(self):
        espec = espec_desde_cnf(RED_MIXTA)
        assert espec.padres == ((1, 2), (0, 1, 2), (0, 1))
        assert not espec.es_binaria()

    def test_las_formulas_escritas_se_conservan(self):
        assert cnf_de_espec(espec_desde_cnf(RED_MIXTA)) == RED_MIXTA

    def test_nodos_deterministas_siguen_binarios(self):
        tpm = tpm_nodo_estado(espec_desde_cnf(RED_MIXTA))
        assert set(np.unique(tpm[:, 0]).tolist()) <= {0.0, 1.0}   # nodo 1
        assert set(np.unique(tpm[:, 2]).tolist()) <= {0.0, 1.0}   # nodo 3
        # Nodo 2 = 0.7·(A∧C) + 0.3·(A∨B). El valor 0.7 aislado es imposible porque
        # A∧C implica A∨B, así que la columna toma {0, 0.3, 1}.
        assert np.unique(np.round(tpm[:, 1], 9)).tolist() == [0.0, 0.3, 1.0]


class TestNoDegeneracionCompartida:
    """`cnf.depende_de_todas` la usan el generador binario y el PBN: vive en `cnf`."""

    def test_detecta_constantes(self):
        assert not cnf.depende_de_todas(np.zeros(4), 2)
        assert not cnf.depende_de_todas(np.ones(4), 2)

    def test_detecta_dependencia_parcial(self):
        # depende del bit 0 pero no del bit 1
        assert not cnf.depende_de_todas(np.array([0.0, 1.0, 0.0, 1.0]), 2)

    def test_vale_para_tablas_de_probabilidades(self):
        assert cnf.depende_de_todas(np.array([0.1, 0.9, 0.3, 0.7]), 2)
