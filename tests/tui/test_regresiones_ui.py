"""Regresiones de la interfaz reportadas al usarla.

Los tres bugs se colaron porque los tests montaban los widgets aislados, donde no
se reproducen: el `Static` vacío, el widget sin hijos y el método con la variable
equivocada sólo fallan al integrarse.
"""

import asyncio

import numpy as np
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select, Static

from src.tui.dataset.screen import DatasetScreen
from src.tui.dataset.widgets import NetworkItem
from src.tui.run.helpers import Programa
from src.tui.run.widgets import OpcionesRow, ProgramCard


@pytest.fixture
def redes(tmp_path, monkeypatch):
    import src.io.fst as fst
    import src.io.manager as manager

    monkeypatch.setattr(manager, "REDES_DIR", tmp_path)
    monkeypatch.setattr(fst, "REDES_DIR", tmp_path)
    espec = fst.generar_espec(4, 2, topologia="anillo", semilla=5)
    np.savetxt(tmp_path / "N4A.csv", fst.tpm_nodo_estado(espec), delimiter=",", fmt="%d")
    fst.guardar_espec("N4B", espec)
    return tmp_path


def _correr(app, cuerpo, size=(160, 45)):
    async def _run():
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            return await cuerpo(app, pilot)

    return asyncio.run(_run())


class _DatasetApp(App):
    def compose(self) -> ComposeResult:
        yield DatasetScreen()


class TestDatasetNoRompeAlSeleccionar:
    """Daba `NameError: name is not defined`: el veredicto de identificabilidad se
    insertó en `__update_display`, que no tiene la variable `name` sino
    `self.selected_network`."""

    def test_click_en_una_red_muestra_el_sistema(self, redes):
        async def cuerpo(app, pilot):
            item = next(i for i in app.query(NetworkItem) if i.network_name == "N4A")
            item.query_one(".net-name").press()
            await pilot.pause()
            await pilot.pause()
            return str(app.query_one("#col-completo", Static).content)

        texto = _correr(_DatasetApp(), cuerpo)
        assert "Error" not in texto, texto[:200]
        assert texto.strip()

    def test_el_veredicto_aparece_en_la_columna(self, redes):
        async def cuerpo(app, pilot):
            item = next(i for i in app.query(NetworkItem) if i.network_name == "N4A")
            item.query_one(".net-name").press()
            await pilot.pause()
            await pilot.pause()
            return str(app.query_one("#col-completo", Static).content)

        texto = _correr(_DatasetApp(), cuerpo)
        assert any(
            v in texto for v in ("IDENTIFICABLE", "DENSA", "especificación")
        ), texto[:200]


class _CardApp(App):
    def __init__(self, programa):
        super().__init__()
        self.programa = programa

    def compose(self) -> ComposeResult:
        yield ProgramCard(self.programa)


class TestFilaDeOpcionesVacia:
    """`Widget.render()` por defecto dibuja el identificador CSS del widget, así que
    una fila sin hijos mostraba literalmente `OpcionesRow.opciones-row` y ocupaba
    alto en la tarjeta."""

    def test_estrategia_sin_opciones_oculta_la_fila(self):
        prog = Programa(nombre="p", estrategia="phi")

        async def cuerpo(app, _):
            fila = app.query_one(OpcionesRow)
            return fila.display, fila.size.height

        display, alto = _correr(_CardApp(prog), cuerpo)
        assert display is False
        assert alto == 0

    def test_estrategia_con_opciones_muestra_la_fila(self):
        prog = Programa(nombre="p", estrategia="qsw_fst")

        async def cuerpo(app, _):
            fila = app.query_one(OpcionesRow)
            return fila.display, len(fila.query(Select))

        display, selects = _correr(_CardApp(prog), cuerpo)
        assert display is True
        assert selects >= 1

    def test_cambiar_a_estrategia_sin_opciones_la_oculta(self):
        prog = Programa(nombre="p", estrategia="qsw_fst")

        async def cuerpo(app, pilot):
            app.query_one("#sel-str-p", Select).value = "phi"
            await pilot.pause()
            return app.query_one(OpcionesRow).display

        assert _correr(_CardApp(prog), cuerpo) is False


class TestIconosConsistentes:
    """El tab Specification usaba `●`/`○` y los otros dos `Ⓢ🅒⚡`: el mismo estado se
    dibujaba distinto según el tab."""

    def test_specification_usa_el_mismo_vocabulario(self, redes):
        from src.tui.spec.screen import SpecScreen
        from src.tui.spec.widgets import SpecItem
        import src.tui.spec.screen as screen

        screen.REDES_DIR = redes

        class _SpecApp(App):
            def compose(self) -> ComposeResult:
                yield SpecScreen()

        async def cuerpo(app, _):
            return {i.spec_name: i.marcas for i in app.query(SpecItem)}

        marcas = _correr(_SpecApp(), cuerpo)
        assert marcas["N4B"] == "Ⓢ"  # sólo spec
        assert set("".join(marcas.values())) <= set("Ⓢ🅒⚡")
