"""Flujo completo spec ↔ csv entre los tabs Specification y Dataset.

Repro del bug reportado al usar la app: materializar un CSV desde Specs,
borrarlo desde Dataset y ver marcas Ⓒ fantasma en Specs. La causa era que cada
tab sólo refrescaba su lista en sus propios eventos; ahora todo tab que muta el
disco llama a `shared/redes.refrescar_tabs_de_redes()`.

Se usa `.press()` y no `pilot.click` sobre los botones de los ítems: el click
por coordenadas depende del layout del tab activo y es indeterminista en tests.
"""

import asyncio

import numpy as np
import pytest
from textual.app import App, ComposeResult
from textual.widgets import Input, Static

from src.io.manager import capacidades, marcas_de_red
from src.tui.dataset.screen import DatasetScreen
from src.tui.dataset.veredicto import VeredictoBar
from src.tui.dataset.widgets import NetworkItem
from src.tui.spec.screen import SpecScreen
from src.tui.spec.widgets import SpecItem


@pytest.fixture
def redes(tmp_path, monkeypatch):
    import src.io.fst as fst
    import src.io.manager as manager
    import src.tui.spec.screen as spec_screen

    # Tres bindings del mismo path: manager y fst lo leen en runtime, pero
    # spec/screen.py lo importó a su namespace (`from src.io.generator import`).
    monkeypatch.setattr(manager, "REDES_DIR", tmp_path)
    monkeypatch.setattr(fst, "REDES_DIR", tmp_path)
    monkeypatch.setattr(spec_screen, "REDES_DIR", tmp_path)
    return tmp_path


class _App(App):
    def compose(self) -> ComposeResult:
        yield SpecScreen()
        yield DatasetScreen()


def _correr(app, cuerpo, size=(160, 90)):
    async def _run():
        async with app.run_test(size=size) as pilot:
            await pilot.pause()
            return await cuerpo(app, pilot)

    return asyncio.run(_run())


def _item(app, nombre):
    return next(i for i in app.query(NetworkItem) if i.network_name == nombre)


def _spec_item(app, nombre):
    return next(i for i in app.query(SpecItem) if i.spec_name == nombre)


def test_ciclo_de_vida_spec_csv(redes):
    """Crear spec → verla en Dataset → materializar → borrar CSV → borrar spec.

    Cada paso afirma el estado que el usuario VE en ambos tabs.
    """

    async def cuerpo(app, pilot):
        ss = app.query_one(SpecScreen)

        # ── 1. Crear spec n=5: aparece en Dataset aunque no tenga CSV ──
        ss.query_one("#inp-n", Input).value = "5"
        ss.query_one("#inp-k", Input).value = "2"
        ss.query_one("#btn-nueva-spec").press()
        await pilot.pause()
        nombre = ss.seleccionada
        assert nombre, "spec no creada"
        assert nombre in [i.network_name for i in app.query(NetworkItem)]
        assert _item(app, nombre).marcas == "Ⓢ"

        # ── 2. Click en Dataset sobre red sólo-spec: mensaje útil, no error crudo ──
        _item(app, nombre).query_one(".net-name").press()
        await pilot.pause()
        await pilot.pause()
        completo = str(app.query_one("#col-completo", Static).content)
        assert "Error" not in completo, completo[:200]
        assert "especificación" in completo  # sólo-spec: instruye materializar
        veredicto = str(app.query_one(VeredictoBar).query_one("#veredicto-text", Static).content)
        assert "tiene especificación" in veredicto

        # ── 3. Materializar: Ⓒ aparece en ambos tabs ──
        ss.query_one("#btn-materializar").press()
        await pilot.pause()
        assert capacidades(nombre)["csv"]
        assert _item(app, nombre).marcas == marcas_de_red(nombre)
        assert _spec_item(app, nombre).marcas == marcas_de_red(nombre)

        # ── 4. Borrar CSV desde Dataset: queda Ⓢ (la spec sigue), sin Ⓒ fantasma ──
        _item(app, nombre).query_one(".net-del").press()
        await pilot.pause()
        assert not capacidades(nombre)["csv"]
        assert nombre in [i.network_name for i in app.query(NetworkItem)]
        assert _item(app, nombre).marcas == "Ⓢ"
        assert _spec_item(app, nombre).marcas == "Ⓢ"  # antes: "ⓈⒸ" con CSV ya borrado

        # ── 5. Borrar la spec: la red desaparece de Dataset ──
        _spec_item(app, nombre).query_one(".spec-del").press()
        await pilot.pause()
        assert not capacidades(nombre)["spec"]
        assert nombre not in [i.network_name for i in app.query(NetworkItem)]
        assert nombre not in [i.spec_name for i in app.query(SpecItem)]

    _correr(_App(), cuerpo)
