"""Testing tab: crear y gestionar patrones de prueba.

Dos vistas:
  - Editar: cuadrantes de generadores (estado/condición/alcance/mecanismo)
  - Preview: tabla de combinaciones generadas
"""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Rule, Select, Static

from src.tui.shared.consts import estilizar
from src.tui.test.generadores import listar_generadores
from src.tui.test.helpers import (
    Patron,
    cargar_patron,
    eliminar_patron,
    formatear_preview,
    guardar_patron,
    listar_patrones,
    siguiente_nombre_patron,
    total_combinaciones,
)
from src.tui.test.widgets import GeneratorItem, PatternItem

CATEGORIAS = ("estados", "condiciones", "alcances", "mecanismos")
TITULOS = {
    "estados": "Estado",
    "condiciones": "Condición",
    "alcances": "Alcance",
    "mecanismos": "Mecanismo",
}


class TestingScreen(Widget):
    """Testing tab: patrones de prueba con generadores de combinaciones."""

    DEFAULT_CSS = estilizar()

    __patron_nombre: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Container(id="test-grid"):
            # ── Columna izquierda: listado de patrones ──
            with VerticalScroll(id="left-pane"):
                yield Label("Patrones", classes="pane-title")
                yield Button("+ Nuevo", variant="primary", id="btn-nuevo-patron")

            # ── Columna derecha ──
            with Vertical(id="right-pane"):
                # Vista 1: Editar cuadrantes
                with Vertical(id="vista-editar"):
                    with Container(id="cuadrantes"):
                        for cat in CATEGORIAS:
                            with Vertical(classes="cuadrante", id=f"cuad-{cat}"):
                                with Horizontal(classes="cuad-header"):
                                    yield Label(TITULOS[cat], classes="cuad-titulo")
                                    yield Select(
                                        options=[(n, n) for n in listar_generadores()],
                                        prompt="Generador...",
                                        id=f"sel-{cat}",
                                    )
                                    yield Button(
                                        "+", variant="primary", id=f"btn-add-{cat}"
                                    )
                                with VerticalScroll(
                                    classes="gen-list", id=f"list-{cat}"
                                ):
                                    pass  # GeneratorItems se agregan dinámicamente

                    yield Rule()
                    with Horizontal(id="footer-editar"):
                        yield Label("Total: 0 combinaciones", id="lbl-total")
                        yield Button("Preview", variant="primary", id="btn-preview")

                # Vista 2: Preview de combinaciones
                with Vertical(id="vista-preview"):
                    with Horizontal(id="preview-toolbar"):
                        yield Label("Dimensiones:", classes="preview-label")
                        yield Input(placeholder="N", id="inp-dims-preview", value="3")
                    with VerticalScroll(id="preview-scroll"):
                        yield Static("", id="preview-contenido")
                    yield Rule()
                    with Horizontal(id="footer-preview"):
                        yield Label("", id="lbl-total-preview")
                        yield Button("Volver", variant="primary", id="btn-volver")

    def on_mount(self) -> None:
        self.__patron_actual: Patron | None = None
        self.__refrescar_patrones()
        self.query_one("#vista-preview").display = False

    # ── Reactive ──

    def watch___patron_nombre(self, viejo: str, nuevo: str) -> None:
        for item in self.query(PatternItem):
            item.add_class(
                "selected"
            ) if item.nombre_patron == nuevo else item.remove_class("selected")

    # ── Eventos ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""

        if bid == "btn-nuevo-patron":
            self.__crear_patron()
        elif bid == "btn-preview":
            self.__mostrar_preview()
        elif bid == "btn-volver":
            self.__mostrar_editar()
        elif bid.startswith("btn-add-"):
            cat = bid.replace("btn-add-", "")
            self.__agregar_generador(cat)

    def on_pattern_item_seleccionado(self, event: PatternItem.Seleccionado) -> None:
        self.__patron_nombre = event.nombre
        self.__cargar_patron(event.nombre)

    def on_pattern_item_eliminado(self, event: PatternItem.Eliminado) -> None:
        eliminar_patron(event.nombre)
        if self.__patron_nombre == event.nombre:
            self.__patron_nombre = ""
            self.__patron_actual = None
            self.__limpiar_cuadrantes()
        self.__refrescar_patrones()

    def on_generator_item_eliminado(self, event: GeneratorItem.Eliminado) -> None:
        if not self.__patron_actual:
            return
        lista = getattr(self.__patron_actual, event.categoria)
        if event.nombre in lista:
            lista.remove(event.nombre)
            guardar_patron(self.__patron_actual)
        self.__refrescar_cuadrante(event.categoria)
        self.__actualizar_total()

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "inp-dims-preview" and self.__patron_actual:
            self.__renderizar_preview()

    # ── Helpers privados ──

    def __crear_patron(self) -> None:
        nombre = siguiente_nombre_patron()
        patron = Patron(nombre=nombre)
        guardar_patron(patron)
        self.__refrescar_patrones()
        self.__patron_nombre = nombre
        self.__cargar_patron(nombre)

    def __refrescar_patrones(self) -> None:
        pane = self.query_one("#left-pane", VerticalScroll)
        for item in list(pane.query(PatternItem)):
            item.remove()
        for nombre in listar_patrones():
            pane.mount(PatternItem(nombre))

    def __cargar_patron(self, nombre: str) -> None:
        try:
            self.__patron_actual = cargar_patron(nombre)
            for cat in CATEGORIAS:
                self.__refrescar_cuadrante(cat)
            self.__actualizar_total()
        except Exception as e:
            self.query_one("#lbl-total", Label).update(f"Error: {e}")

    def __agregar_generador(self, categoria: str) -> None:
        if not self.__patron_actual:
            return
        sel = self.query_one(f"#sel-{categoria}", Select)
        valor = sel.value
        if valor is Select.BLANK:
            return
        nombre_gen = str(valor)
        lista = getattr(self.__patron_actual, categoria)
        if nombre_gen not in lista:
            lista.append(nombre_gen)
            guardar_patron(self.__patron_actual)
            self.__refrescar_cuadrante(categoria)
            self.__actualizar_total()

    def __refrescar_cuadrante(self, categoria: str) -> None:
        contenedor = self.query_one(f"#list-{categoria}", VerticalScroll)
        for item in list(contenedor.query(GeneratorItem)):
            item.remove()
        if not self.__patron_actual:
            return
        for nombre in getattr(self.__patron_actual, categoria):
            contenedor.mount(GeneratorItem(nombre, categoria))

    def __limpiar_cuadrantes(self) -> None:
        for cat in CATEGORIAS:
            self.__refrescar_cuadrante(cat)
        self.__actualizar_total()

    def __actualizar_total(self) -> None:
        if not self.__patron_actual:
            self.query_one("#lbl-total", Label).update("Total: 0 combinaciones")
            return
        total = total_combinaciones(self.__patron_actual)
        self.query_one("#lbl-total", Label).update(f"Total: {total} combinaciones")

    def __mostrar_preview(self) -> None:
        self.query_one("#vista-editar").display = False
        self.query_one("#vista-preview").display = True
        self.__renderizar_preview()

    def __mostrar_editar(self) -> None:
        self.query_one("#vista-preview").display = False
        self.query_one("#vista-editar").display = True

    def __renderizar_preview(self) -> None:
        if not self.__patron_actual:
            self.query_one("#preview-contenido", Static).update(
                "Selecciona un patrón primero"
            )
            return
        try:
            dims_str = self.query_one("#inp-dims-preview", Input).value
            n = int(dims_str) if dims_str else 3
            n = max(1, min(n, 30))
            texto = formatear_preview(self.__patron_actual, n)
            total = total_combinaciones(self.__patron_actual)
            self.query_one("#preview-contenido", Static).update(texto)
            self.query_one("#lbl-total-preview", Label).update(
                f"Total: {total} combinaciones"
            )
        except Exception as e:
            self.query_one("#preview-contenido", Static).update(f"Error: {e}")
