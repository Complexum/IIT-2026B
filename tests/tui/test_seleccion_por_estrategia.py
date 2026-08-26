"""El selector de red se filtra por lo que la estrategia declara necesitar.

Una red es un **nombre** que puede tener CSV, sidecar `.npy` y/o especificación.
Las estrategias con `necesita_espec` corren sobre redes sin CSV —a n grande jamás
va a existir—; las demás lo necesitan. Filtrar en la UI evita ofrecer combinaciones
que fallan recién al ejecutar.
"""

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from src.tui.run.helpers import Programa, redes_compatibles
from src.tui.run.widgets import ProgramCard


class _Harness(App):
    def __init__(self, programa: Programa) -> None:
        super().__init__()
        self.programa = programa

    def compose(self) -> ComposeResult:
        yield ProgramCard(self.programa)


@pytest.fixture
def redes(tmp_path, monkeypatch):
    """Tres redes: sólo CSV, sólo spec, y las dos."""
    import numpy as np

    import src.io.fst as fst
    import src.io.manager as manager

    monkeypatch.setattr(manager, "REDES_DIR", tmp_path)
    monkeypatch.setattr(fst, "REDES_DIR", tmp_path)

    espec = fst.generar_espec(4, 2, topologia="anillo", semilla=5)
    np.savetxt(tmp_path / "N4A.csv", fst.tpm_nodo_estado(espec), delimiter=",", fmt="%d")
    fst.guardar_espec("N4B", espec)
    np.savetxt(tmp_path / "N4C.csv", fst.tpm_nodo_estado(espec), delimiter=",", fmt="%d")
    fst.guardar_espec("N4C", espec)
    return tmp_path


class TestFiltro:
    def test_estrategia_de_tpm_solo_ve_las_que_tienen_csv(self, redes):
        assert {v for _, v in redes_compatibles("analytical")} == {"N4A", "N4C"}

    def test_estrategia_de_spec_solo_ve_las_que_tienen_spec(self, redes):
        assert {v for _, v in redes_compatibles("qsw_fst")} == {"N4B", "N4C"}

    def test_sin_estrategia_se_ven_todas(self, redes):
        assert {v for _, v in redes_compatibles("")} == {"N4A", "N4B", "N4C"}

    def test_la_etiqueta_anota_las_capacidades(self, redes):
        etiquetas = dict((v, e) for e, v in redes_compatibles(""))
        assert etiquetas["N4A"].endswith("🅒")       # sólo csv
        assert etiquetas["N4B"].endswith("Ⓢ")       # sólo spec
        assert etiquetas["N4C"].endswith("Ⓢ🅒")      # las dos


class TestTarjeta:
    def _con_app(self, programa, cuerpo):
        async def _run():
            app = _Harness(programa)
            async with app.run_test() as pilot:
                return await cuerpo(app, pilot)

        return asyncio.run(_run())

    def test_cambiar_a_estrategia_de_spec_refiltra_el_selector(self, redes):
        prog = Programa(
            nombre="p", dataset="N4A", patron="patron-2", estrategia="analytical"
        )

        async def cuerpo(app, pilot):
            app.query_one("#sel-str-p", Select).value = "qsw_fst"
            await pilot.pause()
            sel = app.query_one("#sel-ds-p", Select)
            return {v for _, v in sel._options} if hasattr(sel, "_options") else None

        self._con_app(prog, cuerpo)
        # N4A no tiene spec: la selección deja de ser válida y se limpia.
        assert prog.dataset == ""

    def test_red_valida_para_ambas_sobrevive_al_cambio(self, redes):
        prog = Programa(
            nombre="p", dataset="N4C", patron="patron-2", estrategia="analytical"
        )

        async def cuerpo(app, pilot):
            app.query_one("#sel-str-p", Select).value = "qsw_fst"
            await pilot.pause()

        self._con_app(prog, cuerpo)
        assert prog.dataset == "N4C"
