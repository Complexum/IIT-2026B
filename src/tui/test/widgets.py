"""Widgets del tab Testing: PatternItem y GeneratorItem."""

from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label


class PatternItem(Widget):
    """Entrada de patrón: [nombre clickeable] [✕ eliminar].

    Mensajes que burbujean al padre:
      - PatternItem.Seleccionado(nombre) → on_pattern_item_seleccionado
      - PatternItem.Eliminado(nombre)    → on_pattern_item_eliminado
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
        self.nombre_patron = nombre

    def compose(self) -> ComposeResult:
        yield Button(self.nombre_patron, classes="pat-name")
        yield Button("✕", variant="error", classes="pat-del")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if "pat-del" in event.button.classes:
            self.post_message(self.Eliminado(self.nombre_patron))
        else:
            self.post_message(self.Seleccionado(self.nombre_patron))


class GeneratorItem(Widget):
    """Entrada de generador: [nombre] [✕ eliminar].

    Incluye la categoría (estados/condiciones/alcances/mecanismos)
    para que el padre sepa de qué cuadrante eliminarlo.

    Mensaje: GeneratorItem.Eliminado(nombre, categoria) → on_generator_item_eliminado
    """

    class Eliminado(Message):
        def __init__(self, nombre: str, categoria: str) -> None:
            self.nombre = nombre
            self.categoria = categoria
            super().__init__()

    def __init__(self, nombre: str, categoria: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.nombre_generador = nombre
        self.categoria = categoria

    def compose(self) -> ComposeResult:
        yield Label(self.nombre_generador, classes="gen-name")
        yield Button("✕", variant="error", classes="gen-del")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.Eliminado(self.nombre_generador, self.categoria))
