"""Estrategia `qsw_fst`: QSW sobre la especificación, sin materializar la TPM.

La propiedad que se fija es la **equivalencia** con `analytic` —exhaustivo sobre la
TPM materializada— corriendo ambos sobre la misma red. Si difieren, la ruta nueva
no sirve por más rápida que sea.
"""

import numpy as np
import pytest

from src.iit.core.params import Params
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.runner import importar_estrategias, resolver_estrategia
from src.io.fst import generar_espec, generar_espec_pbn, tpm_nodo_estado
from src.io.manager import reducir_a_subsistema
from src.io.reduccion import reducir_espec

N = 6


def _redes():
    return {
        "binario": generar_espec(N, 2, topologia="anillo", semilla=5),
        "pbn": generar_espec_pbn(N, 3, funciones=3, topologia="anillo", semilla=5),
        "ruido": generar_espec_pbn(N, 3, ruido=0.2, topologia="anillo", semilla=5),
    }


class TestRegistro:
    def test_declara_que_necesita_especificacion(self):
        importar_estrategias()
        assert SIA.necesita_espec.get("qsw_fst") is True
        assert SIA.necesita_mpt.get("qsw_fst") is False

    def test_no_necesita_system(self):
        """El punto de la estrategia: `SubEspec` alcanza, no hace falta un `System`
        (que sería exactamente la materialización que se evita)."""
        espec = _redes()["binario"]
        params = Params("1" * N, "1" * N, "1" * N, "1" * N)
        sol = resolver_estrategia(reducir_espec(espec, params), "qsw_fst")
        assert np.isfinite(sol.perdida)


class TestEquivalenciaConAnalytic:
    @pytest.mark.parametrize("red", ["binario", "pbn", "ruido"])
    @pytest.mark.parametrize("modo", ["estatico", "exacto"])
    def test_misma_perdida_sobre_subsistemas_al_azar(self, red, modo):
        espec = _redes()[red]
        tpm = tpm_nodo_estado(espec)
        rng = np.random.default_rng(3)
        comparados = 0
        for _ in range(20):
            bits = lambda: "".join(str(int(b)) for b in rng.integers(0, 2, size=N))
            params = Params(bits(), bits(), bits(), bits())
            sistema = reducir_a_subsistema(tpm, params)
            if not sistema.dims or not sistema.indices:
                continue
            ref = resolver_estrategia(sistema, "analytical").perdida
            got = resolver_estrategia(
                reducir_espec(espec, params), "qsw_fst", {"modo": modo}
            ).perdida
            assert got == pytest.approx(ref, abs=1e-5), f"params={params}"
            comparados += 1
        assert comparados >= 5


class TestCasosLimite:
    def test_subsistema_vacio_no_rompe(self):
        """Condicionar todo deja el subsistema sin nodos ni dims."""
        espec = _redes()["binario"]
        params = Params("1" * N, "0" * N, "1" * N, "1" * N)
        sol = resolver_estrategia(reducir_espec(espec, params), "qsw_fst")
        assert sol.perdida == pytest.approx(0.0)

    def test_opcion_invalida_lanza(self):
        from src.iit.strategies.python.qsw_fst.code import QSWFst

        with pytest.raises(ValueError, match="inválido"):
            QSWFst.validar_opciones({"modo": "turbo"})

    def test_no_admite_opciones_de_otras_estrategias(self):
        from src.iit.strategies.python.qsw_fst.code import QSWFst

        with pytest.raises(ValueError, match="no admite"):
            QSWFst.validar_opciones({"backend": "c"})
