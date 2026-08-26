"""Generador de FST booleanos (`src/io/fst.py`): la red se define por estructura, no celda por celda.

`generar_red` sortea cada celda de la TPM de forma independiente, lo que produce
una función booleana aleatoria densa: cada nodo depende de las n dimensiones con
influencia 1/2. Estos tests fijan el contrato del generador estructurado, cuya
propiedad central es que la TPM nodo-estado **se deriva** de una especificación
compacta (padres + tabla de verdad por nodo) y sigue siendo consumible por el
pipeline tradicional.
"""

import numpy as np
import pytest

from src.iit.core.params import Params
from src.io.fst import (
    COMPUERTAS,
    EspecFST,
    cargar_espec,
    cnf_de_espec,
    espec_desde_cnf,
    generar_espec,
    guardar_espec,
    marginalizar_a_nodo_estado,
    tpm_estado_estado,
    tpm_nodo_estado,
)
from src.io.manager import preparar_ncubos
from src.iit.strategies.runner import ejecutar

N = 3  # redes chicas: `phi` termina al instante y sirve de verdad de terreno


def _efectivas(columna: np.ndarray, n: int) -> tuple[int, ...]:
    """Dims de las que la columna depende de verdad, según la convención del
    pipeline (`NCube.ndata` reshapea con order='F')."""
    nd = columna.reshape((2,) * n, order="F")
    return tuple(
        j
        for j in range(n)
        if not np.array_equal(np.take(nd, 0, axis=j), np.take(nd, 1, axis=j))
    )


class TestEspecificacion:
    def test_tabla_inconsistente_con_los_padres_lanza(self):
        with pytest.raises(ValueError, match="tabla"):
            EspecFST(n=1, padres=((0,),), tablas=(np.zeros(4),))

    def test_padre_fuera_de_rango_lanza(self):
        with pytest.raises(ValueError, match="padre"):
            EspecFST(n=1, padres=((3,),), tablas=(np.zeros(2),))

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_fan_in_estricto_da_el_fan_in_pedido(self, k):
        """Sin esto `k` miente: una tabla aleatoria puede salir degenerada (con
        k=2, 2 de 16 son constantes y 8 dependen de una sola entrada)."""
        espec = generar_espec(N, k, semilla=7)
        assert espec.fan_in == (k,) * N

    @pytest.mark.parametrize("compuerta", sorted(COMPUERTAS))
    def test_compuertas_con_nombre(self, compuerta):
        espec = generar_espec(N, 2, compuerta=compuerta, semilla=3)
        assert espec.es_binaria()
        assert espec.fan_in == (2,) * N


class TestDerivacionDeLaTPM:
    """La nodo-estado es la marginalización por nodo de la estado-estado."""

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_estado_estado_es_una_distribucion(self, k):
        P = tpm_estado_estado(generar_espec(N, k, semilla=7))
        assert P.shape == (1 << N, 1 << N)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_round_trip_estado_estado_a_nodo_estado(self, k):
        """Es la validación que justifica usar la nodo-estado: no pierde nada
        mientras los nodos sean condicionalmente independientes dado el estado."""
        espec = generar_espec(N, k, semilla=7)
        directa = tpm_nodo_estado(espec)
        via_conjunta = marginalizar_a_nodo_estado(tpm_estado_estado(espec))
        np.testing.assert_allclose(via_conjunta, directa, atol=1e-12)

    def test_red_binaria_da_estado_estado_determinista(self):
        P = tpm_estado_estado(generar_espec(N, 2, semilla=7))
        assert np.isin(P, (0.0, 1.0)).all()
        np.testing.assert_array_equal(P.sum(axis=1), np.ones(1 << N))


class TestConvencionDeBits:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_dims_efectivas_son_exactamente_los_padres(self, k):
        """Prueba de endianness contra el pipeline, no contra el generador: si la
        convención no coincidiera, el conjunto efectivo saldría permutado."""
        espec = generar_espec(N, k, semilla=7)
        tpm = tpm_nodo_estado(espec)
        for cubo in preparar_ncubos(tpm):
            assert _efectivas(cubo.data, N) == espec.padres[cubo.indice]

    def test_tabla_se_indexa_little_endian_por_padre(self):
        """`tablas[i]` bit j ↔ `padres[i][j]`, igual que el índice de estado."""
        espec = EspecFST(
            n=2, padres=((0, 1), (0,)), tablas=(np.array([0.0, 1.0, 0.0, 0.0]), np.zeros(2))
        )
        tpm = tpm_nodo_estado(espec)
        # patrón 1 = (padre0=1, padre1=0) → estado con nodo 0 encendido = índice 1
        assert tpm[0b01, 0] == 1.0
        assert tpm[0b10, 0] == 0.0


class TestAutoriaEnCNF:
    """La red se escribe en CNF; `n`, padres y adyacencia se derivan de ella."""

    RED = {1: [[2, 3]], 2: [[1], [3]], 3: [[1, 2], [-1, -2]]}

    def test_n_y_padres_salen_de_las_formulas(self):
        espec = espec_desde_cnf(self.RED)
        assert espec.n == 3
        assert espec.padres == ((1, 2), (0, 2), (0, 1))

    def test_nodo_faltante_lanza(self):
        with pytest.raises(ValueError, match="faltan los nodos"):
            espec_desde_cnf({1: [[2]], 3: [[1]]})

    def test_la_formula_escrita_se_conserva(self):
        """La CNF canónica es equivalente pero verbosa: `A ∧ C` volvería como
        `(A∨C) ∧ (¬A∨C) ∧ (A∨¬C)`. Un formato de autoría no puede reescribirte."""
        assert cnf_de_espec(espec_desde_cnf(self.RED)) == self.RED

    def test_red_generada_sin_formula_usa_la_canonica(self):
        generada = generar_espec(N, 2, semilla=7)
        assert generada.formulas is None
        recompilada = espec_desde_cnf(cnf_de_espec(generada))
        np.testing.assert_array_equal(
            tpm_nodo_estado(recompilada), tpm_nodo_estado(generada)
        )

    def test_tpm_de_la_red_escrita_a_mano(self):
        """A=B∨C, B=A∧C, C=A xor B evaluada en ABC=010."""
        tpm = tpm_nodo_estado(espec_desde_cnf(self.RED))
        np.testing.assert_array_equal(tpm[0b010], [1.0, 0.0, 1.0])


class TestPersistencia:
    def test_spec_sobrevive_el_viaje_a_disco(self, tmp_path, monkeypatch):
        import src.io.fst as fst

        monkeypatch.setattr(fst, "REDES_DIR", tmp_path)
        espec = generar_espec(N, 2, semilla=7)
        guardar_espec("NTest", espec)
        vuelta = cargar_espec("NTest")
        assert vuelta is not None
        assert vuelta.n == espec.n and vuelta.padres == espec.padres
        for a, b in zip(vuelta.tablas, espec.tablas):
            np.testing.assert_array_equal(a, b)

    def test_red_sin_spec_devuelve_none(self, tmp_path, monkeypatch):
        import src.io.fst as fst

        monkeypatch.setattr(fst, "REDES_DIR", tmp_path)
        assert cargar_espec("NoExiste") is None


class TestPipelineTradicional:
    """El punto del generador: la TPM derivada se ejecuta y coincide con PyPhi."""

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_analytic_qsw_y_phi_coinciden(self, k):
        espec = generar_espec(N, k, semilla=7)
        tpm = tpm_nodo_estado(espec)
        for s in range(1 << N):
            estado = "".join(str((s >> j) & 1) for j in range(N))
            params = Params(estado, "1" * N, "1" * N, "1" * N)
            ref = ejecutar(tpm, params, "phi").perdida
            for estrategia in ("analytical", "qsw"):
                got = ejecutar(tpm, params, estrategia).perdida
                assert got == pytest.approx(ref, abs=1e-5), (
                    f"k={k} estado={estado} {estrategia}={got} phi={ref}"
                )
