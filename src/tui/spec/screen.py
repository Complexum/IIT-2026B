"""Tab Specification: crear e inspeccionar especificaciones de red.

Una **especificación** define la red por su estructura —padres y fórmula por
nodo— en vez de por su TPM materializada. De ella se deriva el CSV de `2^n × n`
de siempre, pero al revés no: una TPM aleatoria densa no admite spec compacta
(ver `io/identificabilidad.py`).

Este tab no ejecuta nada. Crea, inspecciona y —cuando el tamaño lo permite—
materializa el CSV para poder comparar la ruta nueva contra la tradicional sobre
la misma red.
"""

import numpy as np
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static

from src.io.fst import (
    COMPUERTAS,
    cargar_espec,
    generar_espec,
    generar_espec_pbn,
    guardar_espec,
    listar_especs,
    siguiente_nombre_red,
    tpm_nodo_estado,
)
from src.io.generator import REDES_DIR, peso_estimado
from src.io.manager import capacidades, marcas_de_red
from src.io.topologia import MODOS
from src.tui.dataset.modal import DIMS_WARN_THRESHOLD
from src.tui.shared.consts import estilizar
from src.tui.spec.helpers import (
    formatear_adyacencia,
    formatear_diagnostico,
    formatear_formulas,
)
from src.tui.spec.widgets import SpecItem

#: Modelos de nodo. El valor es lo que se le pasa al generador.
MODELOS = {
    "binario": {},
    "PBN L=2": {"funciones": 2},
    "PBN L=3": {"funciones": 3},
    "ruido ε=0.05": {"ruido": 0.05},
    "ruido ε=0.20": {"ruido": 0.20},
}


class SpecScreen(Widget):
    """Crear e inspeccionar especificaciones."""

    DEFAULT_CSS = estilizar()

    seleccionada: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Horizontal(id="spec-toolbar"):
            with Vertical(classes="campo"):
                yield Label("Nodos (n)", classes="campo-label")
                yield Input(placeholder="1-30", id="inp-n")
            with Vertical(classes="campo"):
                yield Label("Fan-in (k)", classes="campo-label")
                yield Input(placeholder="padres/nodo", id="inp-k")
            with Vertical(classes="campo campo-ancho"):
                yield Label("Modelo de nodo", classes="campo-label")
                yield Select(
                    options=[(m, m) for m in MODELOS],
                    value="binario",
                    allow_blank=False,
                    id="sel-modelo",
                )
            with Vertical(classes="campo campo-ancho"):
                yield Label("Topología", classes="campo-label")
                yield Select(
                    options=[(m, m) for m in MODOS],
                    value="anillo",
                    allow_blank=False,
                    id="sel-topologia",
                )
            with Vertical(classes="campo campo-ancho"):
                yield Label("Compuerta", classes="campo-label")
                yield Select(
                    options=[("aleatoria", "aleatoria")]
                    + [(c, c) for c in sorted(COMPUERTAS)],
                    value="aleatoria",
                    allow_blank=False,
                    id="sel-compuerta",
                )
            with Vertical(classes="campo"):
                yield Label(" ", classes="campo-label")
                yield Button("+ Nueva", variant="primary", id="btn-nueva-spec")

        yield Static(
            "[b]n[/b] nodos · [b]k[/b] padres por nodo (fan-in) · "
            "[b]Modelo[/b]: binario = función determinista; PBN L=ℓ = el nodo elige "
            "al azar entre ℓ funciones; ruido ε = una función que se voltea con "
            "probabilidad ε · "
            "[b]Topología[/b]: anillo garantiza que no exista corte cerrado "
            "(si existe, φ = 0 sale trivial) · "
            "[b]Ⓢ[/b] spec  [b]🅒[/b] csv  [b]⚡[/b] npy",
            id="spec-ayuda",
        )

        with Container(id="spec-main"):
            with VerticalScroll(id="spec-lista"):
                yield Label("Especificaciones", classes="pane-title")

            with Vertical(id="spec-panel"):
                with Horizontal(id="spec-acciones"):
                    yield Label("", id="lbl-spec-nombre", classes="pane-title")
                    yield Button(
                        "Materializar CSV", variant="success", id="btn-materializar"
                    )
                with Horizontal(id="spec-cols"):
                    with VerticalScroll(classes="spec-col"):
                        yield Label("Fórmulas", classes="col-title")
                        yield Static("Selecciona una spec ←", id="col-formulas")
                    with VerticalScroll(classes="spec-col"):
                        yield Label("Adyacencia", classes="col-title")
                        yield Static("", id="col-adyacencia")
                    with VerticalScroll(classes="spec-col"):
                        yield Label("Diagnóstico", classes="col-title")
                        yield Static("", id="col-diagnostico")

    def on_mount(self) -> None:
        self.__refrescar_lista()

    # ── Lista ──

    def __refrescar_lista(self) -> None:
        lista = self.query_one("#spec-lista", VerticalScroll)
        for item in lista.query(SpecItem):
            item.remove()
        for nombre in listar_especs():
            lista.mount(SpecItem(nombre, marcas_de_red(nombre)))

    def on_spec_item_selected(self, event: SpecItem.Selected) -> None:
        self.seleccionada = event.name
        self.__mostrar(event.name)

    def on_spec_item_deleted(self, event: SpecItem.Deleted) -> None:
        (REDES_DIR / f"{event.name}.spec.json").unlink(missing_ok=True)
        if self.seleccionada == event.name:
            self.seleccionada = ""
            self.query_one("#col-formulas", Static).update("Selecciona una spec ←")
            self.query_one("#col-adyacencia", Static).update("")
            self.query_one("#col-diagnostico", Static).update("")
            self.query_one("#lbl-spec-nombre", Label).update("")
        self.__refrescar_lista()

    def __mostrar(self, nombre: str) -> None:
        espec = cargar_espec(nombre)
        if espec is None:
            return
        self.query_one("#lbl-spec-nombre", Label).update(nombre)
        self.query_one("#col-formulas", Static).update(formatear_formulas(espec))
        self.query_one("#col-adyacencia", Static).update(formatear_adyacencia(espec))
        self.query_one("#col-diagnostico", Static).update(formatear_diagnostico(espec))

    # ── Acciones ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-nueva-spec":
            self.__crear()
        elif event.button.id == "btn-materializar":
            self.__materializar()

    def __leer_entero(self, selector: str, defecto: int) -> int:
        try:
            return int(self.query_one(selector, Input).value.strip())
        except ValueError:
            return defecto

    def __crear(self) -> None:
        n = self.__leer_entero("#inp-n", 0)
        if not 1 <= n <= 30:
            self.notify("Nodos debe estar entre 1 y 30", severity="error")
            return
        k = max(1, min(self.__leer_entero("#inp-k", 2), n))

        modelo = str(self.query_one("#sel-modelo", Select).value)
        topologia = str(self.query_one("#sel-topologia", Select).value)
        compuerta = str(self.query_one("#sel-compuerta", Select).value)
        compuerta = None if compuerta == "aleatoria" else compuerta

        if topologia == "anillo" and k < 2:
            self.notify("El anillo necesita k ≥ 2", severity="error")
            return

        try:
            kw = MODELOS[modelo]
            if modelo == "binario":
                espec = generar_espec(
                    n, k, compuerta=compuerta, topologia=topologia, semilla=None
                )
            else:
                # El generador PBN sortea sus propias tablas: `compuerta` no aplica.
                espec = generar_espec_pbn(n, k, topologia=topologia, semilla=None, **kw)
        except Exception as e:  # noqa: BLE001 — feedback en la UI, no crash
            self.notify(f"{type(e).__name__}: {e}", severity="error")
            return

        nombre = siguiente_nombre_red(n)
        guardar_espec(nombre, espec)
        self.__refrescar_lista()
        self.seleccionada = nombre
        self.__mostrar(nombre)
        self.notify(f"Creada {nombre} (n={n}, k={k}, {modelo}, {topologia})")

    def __materializar(self) -> None:
        """Escribe el CSV derivado. Es lo que permite correr la ruta tradicional
        sobre la MISMA red y comparar los resultados."""
        nombre = self.seleccionada
        if not nombre:
            self.notify("Selecciona una spec primero", severity="warning")
            return
        espec = cargar_espec(nombre)
        if espec is None:
            return
        if espec.n > DIMS_WARN_THRESHOLD:
            self.notify(
                f"n={espec.n} pesaría ~{peso_estimado(espec.n):.1f} GB; "
                f"materializar está limitado a n ≤ {DIMS_WARN_THRESHOLD}",
                severity="error",
            )
            return

        ruta = REDES_DIR / f"{nombre}.csv"
        tpm = tpm_nodo_estado(espec)
        np.savetxt(
            ruta, tpm, delimiter=",", fmt="%d" if espec.es_binaria() else "%.6f"
        )
        self.__refrescar_lista()
        self.notify(f"{nombre}.csv escrito ({tpm.shape[0]}×{tpm.shape[1]})")
