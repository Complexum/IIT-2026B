"""Reducción a subsistema sobre las tablas locales (`src/io/reduccion.py`).

Es la pieza que faltaba para enchufar el oráculo local al pipeline real: replicar
`manager.reducir_a_subsistema` sin materializar la TPM. La propiedad que se fija es
que el subsistema resultante da el **mismo oráculo**, no uno parecido.
"""

import numpy as np
import pytest

from src.iit.core.params import Params
from src.iit.strategies.python.qsw.oracle import f_cara_batch, preparar_oraculo
from src.iit.strategies.python.zeta_local import (
    f_cara_batch_local,
    preparar_oraculo_local,
)
from src.io import cnf
from src.io.fst import generar_espec, generar_espec_pbn, tpm_nodo_estado
from src.io.manager import reducir_a_subsistema
from src.io.reduccion import (
    SubEspec,
    condicionar_tabla,
    marginalizar_tabla,
    reducir_espec,
)

N = 8


def _redes():
    return {
        "binario": generar_espec(N, 3, semilla=5),
        "pbn": generar_espec_pbn(N, 3, funciones=3, topologia="anillo", semilla=5),
        "ruido": generar_espec_pbn(N, 3, ruido=0.2, semilla=5),
    }


class TestOperacionesSobreLaTabla:
    def test_condicionar_es_un_slice(self):
        padres, tabla = (0, 1), np.array([0.0, 1.0, 2.0, 3.0])
        # bit0 = padre 0, bit1 = padre 1
        assert condicionar_tabla(padres, tabla, 0, 1)[0] == (1,)
        np.testing.assert_array_equal(condicionar_tabla(padres, tabla, 0, 1)[1], [1.0, 3.0])
        np.testing.assert_array_equal(condicionar_tabla(padres, tabla, 1, 0)[1], [0.0, 1.0])

    def test_marginalizar_promedia_las_mitades(self):
        padres, tabla = (0, 1), np.array([0.0, 1.0, 2.0, 3.0])
        ps, out = marginalizar_tabla(padres, tabla, 1)
        assert ps == (0,)
        np.testing.assert_allclose(out, [(0.0 + 2.0) / 2, (1.0 + 3.0) / 2])

    def test_dim_ausente_no_cambia_nada(self):
        padres, tabla = (0, 2), np.array([0.0, 1.0, 2.0, 3.0])
        assert condicionar_tabla(padres, tabla, 1, 1) == (padres, tabla)
        assert marginalizar_tabla(padres, tabla, 1) == (padres, tabla)

    def test_condicionar_coincide_con_propagacion_unitaria(self):
        """Sobre la fórmula, condicionar es el paso de Davis-Putnam. Las dos rutas
        —tabla y CNF— tienen que dar lo mismo."""
        formula = [[1, 2], [-1, -2]]  # A xor B
        padres = cnf.variables_de(formula)
        tabla = cnf.evaluar(formula, padres)
        for valor in (0, 1):
            via_cnf = cnf.condicionar(formula, 1, valor)  # variable 1 = nodo 0
            ps_cnf = cnf.variables_de(via_cnf)
            _, via_tabla = condicionar_tabla(padres, tabla, 0, valor)
            np.testing.assert_array_equal(cnf.evaluar(via_cnf, ps_cnf), via_tabla)


class TestSubEspec:
    def test_estado_debe_cubrir_las_dims(self):
        with pytest.raises(ValueError, match="una entrada por dim"):
            SubEspec(dims=(0, 1), indices=(0,), padres=((0,),), tablas=(np.zeros(2),), estado=(0,))

    def test_tabla_inconsistente_lanza(self):
        with pytest.raises(ValueError, match="tabla"):
            SubEspec(dims=(0,), indices=(0,), padres=((0,),), tablas=(np.zeros(4),), estado=(0,))


class TestEquivaleAlPipeline:
    """Lo que importa: mismas etiquetas y mismo oráculo que `reducir_a_subsistema`."""

    @pytest.mark.parametrize("red", ["binario", "pbn", "ruido"])
    def test_dims_e_indices_coinciden(self, red):
        espec = _redes()[red]
        rng = np.random.default_rng(3)
        for _ in range(25):
            bits = lambda: "".join(str(int(b)) for b in rng.integers(0, 2, size=N))
            params = Params(bits(), bits(), bits(), bits())
            sis = reducir_a_subsistema(tpm_nodo_estado(espec), params)
            sub = reducir_espec(espec, params)
            assert sub.dims == tuple(sis.dims)
            assert sub.indices == tuple(c.indice for c in sis.ncubos)

    @pytest.mark.parametrize("red", ["binario", "pbn", "ruido"])
    def test_mismo_oraculo_sobre_todos_los_cortes(self, red):
        espec = _redes()[red]
        rng = np.random.default_rng(3)
        comparados = 0
        for _ in range(25):
            bits = lambda: "".join(str(int(b)) for b in rng.integers(0, 2, size=N))
            params = Params(bits(), bits(), bits(), bits())
            sis = reducir_a_subsistema(tpm_nodo_estado(espec), params)
            D, M = len(sis.dims), len(sis.indices)
            if D == 0 or M == 0 or D + M > 16:
                continue
            sub = reducir_espec(espec, params)

            todas = np.arange(1 << (D + M), dtype=np.int64)
            mec = todas & ((1 << D) - 1)
            alc = ((todas[:, None] >> (D + np.arange(M))) & 1).astype(bool)

            ref = f_cara_batch(preparar_oraculo(sis), mec, alc).astype(np.float64)
            loc = f_cara_batch_local(preparar_oraculo_local(sub, sub.estado), mec, alc)
            # La referencia acumula en float32; el oráculo local en float64.
            np.testing.assert_allclose(loc, ref, atol=1e-6)
            comparados += 1
        assert comparados >= 5, "muy pocos subsistemas no triviales para concluir"

    def test_sin_reduccion_es_el_sistema_completo(self):
        espec = _redes()["binario"]
        params = Params("1" * N, "1" * N, "1" * N, "1" * N)
        sub = reducir_espec(espec, params)
        assert sub.dims == tuple(range(N))
        assert sub.indices == tuple(range(N))
        assert sub.padres == espec.padres
