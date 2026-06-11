"""Results tab: visualizar soluciones guardadas en data/output/.

Cada CSV contiene filas con campos:
    indice, estado, condicion, alcance, mecanismo, perdida, tiempo, particion, plataforma
"""

from pathlib import Path

from rich import box as rich_box
from rich.table import Table
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Static

from src.io.manager import cargar_resultados, listar_resultados
from src.tui.results.widgets import ResultItem
from src.tui.run.helpers import cargar_programa, extract_program_name
from src.tui.shared.consts import estilizar

RESULTADOS_DIR = Path("data/output")


class _InvHScroll(ScrollableContainer):
    """ScrollableContainer con dirección de scroll horizontal invertida."""

    def scroll_left(self, *a, **kw) -> None:
        super().scroll_right(*a, **kw)

    def scroll_right(self, *a, **kw) -> None:
        super().scroll_left(*a, **kw)


class ResultsScreen(Widget):
    """Results tab: explorar resultados de ejecuciones pasadas."""

    DEFAULT_CSS = estilizar()

    __resultado_nombre: reactive[str] = reactive("")

    def compose(self) -> ComposeResult:
        with Container(id="results-grid"):
            # Columna izquierda: listado de resultados
            with VerticalScroll(id="left-pane"):
                yield Label("Programas", classes="pane-title")

            # Columna derecha: detalle de resultados
            with _InvHScroll(id="right-pane"):
                yield Label(
                    "Selecciona un resultado ←",
                    id="result-title",
                    classes="content-title",
                )
                yield Static("", id="result-display")

    def on_mount(self) -> None:
        self.__refrescar_resultados()

    # ── Reactive ──

    def watch___resultado_nombre(self, viejo: str, nuevo: str) -> None:
        for item in self.query(ResultItem):
            item.add_class(
                "selected"
            ) if item.nombre_resultado == nuevo else item.remove_class("selected")

    # ── Eventos ──

    def on_result_item_seleccionado(self, event: ResultItem.Seleccionado) -> None:
        self.__resultado_nombre = event.nombre
        self.__mostrar_detalle(event.nombre)

    def on_result_item_eliminado(self, event: ResultItem.Eliminado) -> None:
        ruta = RESULTADOS_DIR / f"{event.nombre}.csv"
        if ruta.exists():
            ruta.unlink()
        meta = RESULTADOS_DIR / f"{event.nombre}.meta.json"
        if meta.exists():
            meta.unlink()
        try:
            ruta.parent.rmdir()
        except OSError:
            pass
        if self.__resultado_nombre == event.nombre:
            self.__resultado_nombre = ""
            self.query_one("#result-display", Static).update("")
            self.query_one("#result-title", Label).update("Selecciona un resultado ←")
        self.__refrescar_resultados()

        # Notificar a Analysis para que también refresque su lista
        from src.tui.analysis.screen import AnalysisScreen

        try:
            self.app.query_one(AnalysisScreen).refrescar()
        except Exception:
            pass

    # ── API pública ──

    def refrescar(self) -> None:
        """Recargar la lista de resultados (llamable desde otros widgets)."""
        self.__refrescar_resultados()

    # ── Helpers privados ──

    def __refrescar_resultados(self) -> None:
        pane = self.query_one("#left-pane", VerticalScroll)
        for item in list(pane.query(ResultItem)):
            item.remove()
        for nombre in listar_resultados():
            pane.mount(ResultItem(nombre))

    def __mostrar_detalle(self, nombre: str) -> None:
        """Parsear CSV y mostrar como tabla Rich."""
        try:
            contenido = cargar_resultados(nombre)
            lineas = contenido.strip().split("\n")

            estrategia = ""
            try:
                prog_name = extract_program_name(nombre)
                estrategia = cargar_programa(prog_name).estrategia
            except Exception:
                pass
            titulo = f"Resultado: {nombre}" + (
                f"  [{estrategia}]" if estrategia else ""
            )
            self.query_one("#result-title", Label).update(titulo)

            display = self.query_one("#result-display", Static)

            if not lineas:
                display.update("Archivo vacío")
                return

            encabezados = self.__parsear_linea_csv(lineas[0])
            filas = [
                self.__parsear_linea_csv(line) for line in lineas[1:] if line.strip()
            ]

            if not filas:
                display.update("Sin soluciones")
                return

            # Estimar ancho necesario desde los datos para habilitar scroll horizontal
            col_widths = [len(h) for h in encabezados]
            for valores in filas[:20]:
                for j, v in enumerate(valores):
                    first_line = v.split("\\n")[0] if "\\n" in v else v
                    col_widths[j] = max(col_widths[j], len(first_line))
            ancho = sum(col_widths) + len(encabezados) * 4 + 4
            display.styles.width = max(ancho, 160)

            tabla = Table(
                show_header=True,
                header_style="bold cyan",
                show_lines=False,
                box=rich_box.SIMPLE,
                row_styles=["", "on grey11"],
                padding=(0, 1),
                expand=True,
            )
            for h in encabezados:
                if h == "particion":
                    tabla.add_column(h, min_width=24, no_wrap=False)
                else:
                    tabla.add_column(h, no_wrap=True)

            for valores in filas:
                fila: list[str] = []
                for h, v in zip(encabezados, valores):
                    if h == "particion":
                        fila.append(v.replace("\\n", "\n"))
                    elif h == "tiempo_wall_s":
                        try:
                            fila.append(f"{float(v):.4f}s")
                        except ValueError:
                            fila.append(v)
                    else:
                        fila.append(v)
                tabla.add_row(*fila)

            display.update(tabla)

        except Exception as e:
            self.query_one("#result-title", Label).update(f"Error: {nombre}: {e}")

    def __parsear_linea_csv(self, linea: str) -> list[str]:
        """Parser CSV simple que maneja campos entrecomillados."""
        campos: list[str] = []
        actual = ""
        en_comillas = False

        for c in linea:
            if c == '"':
                en_comillas = not en_comillas
            elif c == "," and not en_comillas:
                campos.append(actual.strip())
                actual = ""
            else:
                actual += c

        campos.append(actual.strip())
        return campos
