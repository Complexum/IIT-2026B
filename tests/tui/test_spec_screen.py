"""Tab Specification: crear e inspeccionar especificaciones.

La información que más importa del panel es la **separabilidad**: una red separable
admite un corte cerrado que da `φ = 0` sin importar los valores de las tablas, y
conviene enterarse al crearla y no después de un barrido.
"""

import asyncio
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Select, Static

from src.tui.spec.screen import SpecScreen
from src.tui.spec.widgets import SpecItem


class _Harness(App):
    def compose(self) -> ComposeResult:
        yield SpecScreen()


@pytest.fixture
def redes_temp(tmp_path, monkeypatch):
    """Aísla `REDES_DIR` en los tres módulos que lo importaron por valor."""
    import src.io.fst as fst
    import src.io.generator as generator
    import src.tui.spec.screen as screen

    for modulo in (fst, generator, screen):
        monkeypatch.setattr(modulo, "REDES_DIR", tmp_path)
    return tmp_path


def _con_app(cuerpo):
    async def _run():
        app = _Harness()
        async with app.run_test() as pilot:
            return await cuerpo(app, pilot)

    return asyncio.run(_run())


async def _crear(app, pilot, n: int, k: int, modelo: str, topologia: str):
    app.query_one("#inp-n", Input).value = str(n)
    app.query_one("#inp-k", Input).value = str(k)
    app.query_one("#sel-modelo", Select).value = modelo
    app.query_one("#sel-topologia", Select).value = topologia
    await pilot.pause()
    app.query_one("#btn-nueva-spec", Button).press()
    await pilot.pause()


class TestCreacion:
    def test_crea_y_lista_la_spec(self, redes_temp: Path):
        async def cuerpo(app, pilot):
            await _crear(app, pilot, 5, 2, "binario", "anillo")
            return [i.spec_name for i in app.query(SpecItem)]

        assert _con_app(cuerpo) == ["N5A"]
        assert (redes_temp / "N5A.spec.json").exists()

    def test_no_escribe_csv_al_crear(self, redes_temp: Path):
        """Una spec de n grande no puede materializarse; crear no debe intentarlo."""

        async def cuerpo(app, pilot):
            await _crear(app, pilot, 5, 2, "binario", "anillo")

        _con_app(cuerpo)
        assert not (redes_temp / "N5A.csv").exists()

    def test_rechaza_n_invalido(self, redes_temp: Path):
        async def cuerpo(app, pilot):
            await _crear(app, pilot, 99, 2, "binario", "anillo")
            return len(app.query(SpecItem))

        assert _con_app(cuerpo) == 0


class TestPanelDeInspeccion:
    def test_anillo_se_reporta_no_separable(self, redes_temp: Path):
        async def cuerpo(app, pilot):
            await _crear(app, pilot, 10, 3, "PBN L=3", "anillo")
            return app.query_one("#col-diagnostico", Static).content

        assert "no separable" in str(_con_app(cuerpo))

    def test_aleatoria_con_k_chico_se_reporta_separable(self, redes_temp: Path):
        """Es el aviso que anticipa un φ = 0 trivial."""

        async def cuerpo(app, pilot):
            await _crear(app, pilot, 12, 2, "binario", "aleatoria")
            return str(app.query_one("#col-diagnostico", Static).content)

        texto = _con_app(cuerpo)
        assert "SEPARABLE" in texto and "corte cerrado" in texto

    def test_muestra_formulas_y_adyacencia(self, redes_temp: Path):
        async def cuerpo(app, pilot):
            await _crear(app, pilot, 4, 2, "binario", "anillo")
            return (
                str(app.query_one("#col-formulas", Static).content),
                str(app.query_one("#col-adyacencia", Static).content),
            )

        formulas, adyacencia = _con_app(cuerpo)
        assert formulas.count("\n") == 3  # una línea por nodo
        assert "A" in formulas and "D" in formulas
        assert "1" in adyacencia  # hay al menos una arista

    def test_pbn_se_reporta_probabilistica(self, redes_temp: Path):
        async def cuerpo(app, pilot):
            await _crear(app, pilot, 6, 3, "ruido ε=0.20", "anillo")
            return str(app.query_one("#col-diagnostico", Static).content)

        assert "probabilística" in _con_app(cuerpo)


class TestMaterializar:
    def test_escribe_el_csv_derivado(self, redes_temp: Path):
        """Es lo que permite correr la ruta tradicional sobre la MISMA red."""

        async def cuerpo(app, pilot):
            await _crear(app, pilot, 5, 2, "binario", "anillo")
            app.query_one("#btn-materializar", Button).press()
            await pilot.pause()

        _con_app(cuerpo)
        csv = redes_temp / "N5A.csv"
        assert csv.exists()
        assert len(csv.read_text().strip().splitlines()) == 2**5

    def test_sin_seleccion_no_hace_nada(self, redes_temp: Path):
        async def cuerpo(app, pilot):
            app.query_one("#btn-materializar", Button).press()
            await pilot.pause()

        _con_app(cuerpo)
        assert list(redes_temp.glob("*.csv")) == []


class TestMontajeEnLaApp:
    """El tab salía **en blanco**: los tests montaban `SpecScreen` como widget raíz,
    donde siempre ocupa la pantalla entera, así que no detectaban que dentro de un
    `TabPane` colapsaba a altura cero por faltarle la regla `SpecScreen { height:
    1fr }` en su hoja de estilos.
    """

    def test_ocupa_altura_dentro_del_tab(self):
        from src.tui.app import MIPApp

        async def _run():
            app = MIPApp()
            async with app.run_test(size=(160, 45)) as pilot:
                await pilot.pause()
                pantalla = app.query_one(SpecScreen)
                return {
                    "pantalla": pantalla.size,
                    "toolbar": app.query_one("#spec-toolbar").size,
                    "lista": app.query_one("#spec-lista").size,
                    "panel": app.query_one("#spec-panel").size,
                }

        tam = asyncio.run(_run())
        for nombre, size in tam.items():
            assert size.height > 0, f"{nombre} tiene altura 0"
            assert size.width > 0, f"{nombre} tiene ancho 0"

    def test_los_controles_estan_presentes(self):
        from textual.widgets import Button, Input, Select

        from src.tui.app import MIPApp

        async def _run():
            app = MIPApp()
            async with app.run_test(size=(160, 45)) as pilot:
                await pilot.pause()
                pantalla = app.query_one(SpecScreen)
                return (
                    len(pantalla.query(Input)),
                    len(pantalla.query(Select)),
                    {b.id for b in pantalla.query(Button) if b.id},
                )

        inputs, selects, botones = asyncio.run(_run())
        assert inputs == 2  # n y k
        assert selects == 3  # modelo, topología, compuerta
        assert {"btn-nueva-spec", "btn-materializar"} <= botones
