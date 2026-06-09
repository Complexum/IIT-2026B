"""Modal de confirmación para carga de redes computacionalmente costosas."""

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Label

from src.tui.shared.consts import estilizar

DIMS_WARN_THRESHOLD = 10  # 2^10 = 1024 filas — redes mayores pueden tardar minutos


class ConfirmarCargaModal(ModalScreen[bool]):
    """Confirmación antes de cargar una red con muchas dimensiones."""

    DEFAULT_CSS = estilizar()

    def __init__(self, nombre: str, n_dims: int) -> None:
        super().__init__()
        self._nombre = nombre
        self._n_dims = n_dims

    def compose(self) -> ComposeResult:
        with Container():
            yield Label(
                f"⚠  Red [bold]{self._nombre}[/bold] — "
                f"{self._n_dims} dimensiones ({2**self._n_dims:,} estados)"
            )
            yield Label("Cargar puede tomar decenas de minutos según el equipo.")
            yield Label("¿Continuar?")
            with Horizontal(classes="modal-btns"):
                yield Button("Sí, cargar", variant="warning", id="btn-si")
                yield Button("Cancelar", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-si")
