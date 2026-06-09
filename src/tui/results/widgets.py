"""Widgets del tab Results: ResultItem."""

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Static


class ResultItem(Widget):
    """Entrada de resultado: [nombre clickeable] [✕ eliminar].

    Mensajes:
      - ResultItem.Seleccionado(nombre) → on_result_item_seleccionado
      - ResultItem.Eliminado(nombre)    → on_result_item_eliminado
    """

    class Seleccionado(Message):
        def __init__(self, nombre: str) -> None:
            self.nombre = nombre
            super().__init__()

    class Eliminado(Message):
        def __init__(self, nombre: str) -> None:
            self.nombre = nombre
            super().__init__()

    def __init__(self, nombre: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nombre_resultado = nombre

    def compose(self) -> ComposeResult:
        # Static wraps text; Button truncates with ellipsis
        yield Static(self.nombre_resultado, classes="res-name")
        yield Button("✕", variant="error", classes="res-del")

    def on_click(self) -> None:
        """Click anywhere on the item selects it."""
        # Only select if not clicking the delete button
        # (delete button handles its own press)
        self.post_message(self.Seleccionado(self.nombre_resultado))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if "res-del" in event.button.classes:
            self.post_message(self.Eliminado(self.nombre_resultado))
