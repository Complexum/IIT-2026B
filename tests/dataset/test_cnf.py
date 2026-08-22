"""Fórmulas CNF (`src/io/cnf.py`): el formato de autoría de las funciones de nodo.

La tabla de verdad es el compilado; la CNF es lo que se escribe y lo que se guarda.
Estos tests fijan la convención DIMACS (variable `v` ↔ nodo `v-1`, negativo =
negado, cláusula = OR, fórmula = AND) y el ida y vuelta con la tabla.
"""

import numpy as np
import pytest

from src.io import cnf

# A=1, B=2, C=3 — la red del ejemplo: A=B∨C, B=A∧C, C=A xor B
RED = {1: [[2, 3]], 2: [[1], [3]], 3: [[1, 2], [-1, -2]]}


class TestConvencion:
    def test_variables_se_infieren_de_la_formula(self):
        """Los padres no se declaran aparte: salen de lo que la fórmula menciona."""
        assert cnf.variables_de(RED[1]) == (1, 2)  # B, C → nodos 1 y 2
        assert cnf.variables_de(RED[2]) == (0, 2)
        assert cnf.variables_de(RED[3]) == (0, 1)

    @pytest.mark.parametrize(
        "clausulas, esperado",
        [
            ([[2, 3]], [0, 1, 1, 1]),               # B ∨ C
            ([[1], [3]], [0, 0, 0, 1]),             # A ∧ C
            ([[1, 2], [-1, -2]], [0, 1, 1, 0]),     # A xor B
            ([[-1, -2]], [1, 1, 1, 0]),             # ¬(A ∧ B) = NAND
        ],
    )
    def test_evaluar_compila_la_tabla(self, clausulas, esperado):
        padres = cnf.variables_de(clausulas)
        np.testing.assert_array_equal(cnf.evaluar(clausulas, padres), esperado)

    def test_literal_negativo_niega(self):
        assert cnf.evaluar([[1]], (0,)).tolist() == [0.0, 1.0]
        assert cnf.evaluar([[-1]], (0,)).tolist() == [1.0, 0.0]

    def test_bit_j_del_indice_es_el_padre_j(self):
        """Little-endian sobre `padres`, igual que el índice de estado."""
        tabla = cnf.evaluar([[1]], (0, 1))  # depende sólo del primer padre
        assert tabla[0b01] == 1.0  # padre 0 encendido
        assert tabla[0b10] == 0.0  # padre 1 encendido, padre 0 apagado


class TestCasosLimite:
    def test_and_vacio_es_constante_uno(self):
        assert cnf.evaluar([], ()).tolist() == [1.0]

    def test_clausula_vacia_es_constante_cero(self):
        assert cnf.evaluar([[]], ()).tolist() == [0.0]

    def test_literal_cero_lanza(self):
        with pytest.raises(ValueError, match="literal 0"):
            cnf.validar([[1, 0]], 3)

    def test_variable_fuera_de_rango_lanza(self):
        with pytest.raises(ValueError, match="fuera de rango"):
            cnf.validar([[4]], 3)


class TestIdaYVuelta:
    @pytest.mark.parametrize("v", sorted(RED))
    def test_tabla_a_cnf_a_tabla(self, v):
        padres = cnf.variables_de(RED[v])
        tabla = cnf.evaluar(RED[v], padres)
        canonica = cnf.desde_tabla(tabla, padres)
        np.testing.assert_array_equal(cnf.evaluar(canonica, padres), tabla)

    def test_desde_tabla_de_constantes(self):
        assert cnf.desde_tabla(np.ones(4), (0, 1)) == []
        assert cnf.desde_tabla(np.zeros(4), (0, 1)) == [[]]

    def test_desde_tabla_rechaza_no_binarias(self):
        with pytest.raises(ValueError, match="binarias"):
            cnf.desde_tabla(np.array([0.0, 0.5]), (0,))

    def test_formatear_es_legible(self):
        assert cnf.formatear(RED[3], ["A", "B", "C"]) == "⟨A + B⟩⟨¬A + ¬B⟩"
        assert cnf.formatear(RED[2], ["A", "B", "C"]) == "AC"  # cláusulas unitarias
        assert cnf.formatear([], None) == "1"
        assert cnf.formatear([[]], None) == "0"


class TestNombresDeVariable:
    """El alfabeto se agota a 26 y el oráculo local llega a 60+ nodos.

    Se usa base-26 estilo columna de hoja de cálculo en vez del alfabeto griego,
    que acá colisiona con notación ya tomada (`δ = H − p`, `φ` la cantidad
    calculada, `Σ`, `Θ`, `σ`, `ε`) y encima se agotaría a 50.
    """

    @pytest.mark.parametrize(
        "indice, esperado",
        [
            (0, "A"),
            (1, "B"),
            (25, "Z"),
            (26, "AA"),
            (27, "AB"),
            (51, "AZ"),
            (52, "BA"),
            (701, "ZZ"),
            (702, "AAA"),
        ],
    )
    def test_base_26(self, indice, esperado):
        assert cnf.nombre_de_variable(indice) == esperado

    def test_el_nombre_es_el_numero_de_variable_dimacs(self):
        """`.spec.json` numera 1-indexado y las columnas también: `A=1`, `Z=26`,
        `AA=27`. La correspondencia no es casual, es la misma indexación."""
        for indice in range(60):
            variable = indice + 1
            # decodificar el nombre de vuelta a número debe dar la variable
            valor = 0
            for ch in cnf.nombre_de_variable(indice):
                valor = valor * 26 + (ord(ch) - ord("A") + 1)
            assert valor == variable

    def test_no_se_repiten_ni_se_agotan(self):
        nombres = cnf.nombres_de_variables(1000)
        assert len(set(nombres)) == 1000

    def test_solo_ascii_mayusculas(self):
        assert all(n.isascii() and n.isupper() for n in cnf.nombres_de_variables(200))
