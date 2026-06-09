"""Tests para el flujo de ejecución: ejecutar() end-to-end sin TUI."""

import pytest

from src.iit.core.params import Params
from src.iit.strategies.runner import ejecutar
from src.io.manager import cargar_mpt
from src.tui.test.helpers import cargar_patron, generar_combinaciones


class TestEjecucion:

    def test_ejecutar_basic_n3a(self):
        tpm = cargar_mpt("N3A")
        params = Params("100", "111", "110", "011")
        sol = ejecutar(tpm, params, "basic")
        assert sol is not None
        assert sol.perdida >= 0.0
        assert sol.particion != ""

    def test_ejecutar_strategy_desconocida(self):
        tpm = cargar_mpt("N3A")
        params = Params("100", "111", "110", "011")
        with pytest.raises(ValueError):
            ejecutar(tpm, params, "no_existe")

    def test_generar_combinaciones_patron1(self):
        patron = cargar_patron("patron-1")
        combis = generar_combinaciones(patron, 3)
        assert len(combis) > 0
        for est, cond, alc, mec in combis:
            assert len(est) == 3
            assert set(est) <= {"0", "1"}

    def test_params_validos_desde_combinacion(self):
        """Cada combo del patrón produce Params sin excepción."""
        patron = cargar_patron("patron-1")
        tpm = cargar_mpt("N3A")
        n = tpm.shape[1]
        combis = generar_combinaciones(patron, n)
        for est, cond, alc, mec in combis:
            p = Params(est, cond, alc, mec)
            assert p is not None

    def test_ejecutar_retorna_perdida_finita(self):
        """perdida no puede ser inf ni NaN."""
        import math
        tpm = cargar_mpt("N3A")
        params = Params("100", "111", "110", "011")
        sol = ejecutar(tpm, params, "basic")
        assert math.isfinite(sol.perdida)
