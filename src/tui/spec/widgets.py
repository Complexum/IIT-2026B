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

    def __init__(self, name: str, materializada: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self.spec_name = name
        self.materializada = materializada

    def compose(self) -> ComposeResult:
        # ● = tiene CSV materializado, ○ = sólo spec
        marca = "●" if self.materializada else "○"
        yield Button(f"{marca} {self.spec_name}", classes="spec-name")
        yield Button("✕", variant="error", classes="spec-del")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if "spec-del" in event.button.classes:
            self.post_message(self.Deleted(self.spec_name))
        else:
            self.post_message(self.Selected(self.spec_name))
