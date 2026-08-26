"""Fila de opciones de estrategia en `ProgramCard`.

`SIA.opciones` documenta desde siempre que "la TUI renderiza un selector por
opción", pero sólo el CLI las aceptaba (`--opcion attr=valor`). Estos tests fijan
el comportamiento de esa fila.
"""

import asyncio

from textual.app import App, ComposeResult
from textual.widgets import Select

from src.tui.run.helpers import Programa, opciones_de_estrategia
from src.tui.run.widgets import OpcionesRow, ProgramCard


class _Harness(App):
    def __init__(self, programa: Programa) -> None:
        super().__init__()
        self.programa = programa

    def compose(self) -> ComposeResult:
        yield ProgramCard(self.programa)


def _con_app(programa: Programa, cuerpo):
    """Monta la app en un event loop propio y corre `cuerpo(app, pilot)`.

    Se usa `asyncio.run` en vez de `pytest.mark.asyncio` para no agregar
    `pytest-asyncio` como dependencia sólo por estos tests.
    """

    async def _run():
        app = _Harness(programa)
        async with app.run_test() as pilot:
            return await cuerpo(app, pilot)

    return asyncio.run(_run())


def _prog(estrategia: str, opciones: dict | None = None) -> Programa:
    return Programa(
        nombre="test-prog",
        dataset="N4A",
        patron="patron-1",
        estrategia=estrategia,
        opciones=dict(opciones or {}),
    )


class TestHelper:
    def test_devuelve_las_opciones_declaradas(self):
        assert set(opciones_de_estrategia("qsw")) == {"backend", "modo", "k"}

    def test_estrategia_sin_opciones_da_vacio(self):
        assert opciones_de_estrategia("analytical") == {}

    def test_estrategia_inexistente_no_lanza(self):
        assert opciones_de_estrategia("no-existe") == {}


def _select_de(app, attr: str) -> Select:
    return next(s for s in app.query_one(OpcionesRow).query(Select) if s.name == attr)


class TestRenderizado:
    def test_estrategia_configurable_dibuja_un_select_por_opcion(self):
        async def cuerpo(app, _):
            return {s.name for s in app.query_one(OpcionesRow).query(Select)}

        assert _con_app(_prog("qsw"), cuerpo) == {"backend", "modo", "k"}

    def test_estrategia_sin_opciones_no_dibuja_nada(self):
        async def cuerpo(app, _):
            return len(app.query_one(OpcionesRow).query(Select))

        assert _con_app(_prog("analytical"), cuerpo) == 0

    def test_muestra_el_valor_guardado_no_el_default(self):
        async def cuerpo(app, _):
            return _select_de(app, "modo").value

        assert _con_app(_prog("qsw", {"modo": "estocastico"}), cuerpo) == "estocastico"

    def test_valor_invalido_guardado_cae_al_default(self):
        """Un JSON viejo con un valor que la estrategia ya no admite no debe
        romper el render."""

        async def cuerpo(app, _):
            return _select_de(app, "modo").value

        assert _con_app(_prog("qsw", {"modo": "turbo"}), cuerpo) == (
            opciones_de_estrategia("qsw")["modo"][0]
        )


class TestInteraccion:
    def test_cambiar_una_opcion_la_guarda_en_el_programa(self):
        prog = _prog("qsw")

        async def cuerpo(app, pilot):
            _select_de(app, "modo").value = "estocastico"
            await pilot.pause()

        _con_app(prog, cuerpo)
        assert prog.opciones["modo"] == "estocastico"

    def test_cambiar_de_estrategia_descarta_lo_que_la_nueva_no_admite(self):
        """Un programa que fue `qsw` con `backend=c` reventaba al pasarlo a una
        estrategia sin esa opción. `podar_opciones` lo evita; acá se verifica que
        además la fila se redibuja."""
        prog = _prog("qsw", {"backend": "c", "modo": "estatico"})

        async def cuerpo(app, pilot):
            app.query_one("#sel-str-test-prog", Select).value = "queyranne"
            await pilot.pause()
            return len(app.query_one(OpcionesRow).query(Select))

        assert _con_app(prog, cuerpo) == 0
        assert "backend" not in prog.opciones
