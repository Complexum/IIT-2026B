from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button


class SpecItem(Widget):
    """Entrada de la lista de especificaciones: nombre clickeable + borrar."""

    class Selected(Message):
        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    class Deleted(Message):
        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    def __init__(self, name: str, marcas: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.spec_name = name
        #: Mismo vocabulario que Dataset y Execution: `Ⓢ` spec, `Ⓒ` csv, `Ⓝ` npy.
        #: Antes usaba `●`/`○`, distinto de los otros tabs y por eso ilegible.
        self.marcas = marcas

    def compose(self) -> ComposeResult:
        yield Button(f"{self.spec_name}  {self.marcas}".rstrip(), classes="spec-name")
        yield Button("✕", variant="error", classes="spec-del")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if "spec-del" in event.button.classes:
            self.post_message(self.Deleted(self.spec_name))
        else:
            self.post_message(self.Selected(self.spec_name))
