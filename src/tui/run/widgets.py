"""Widgets del tab Execution: ProgramCard."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Label, ProgressBar, Rule, Select

from pathlib import Path

from src.io.manager import listar_redes

_OUTPUT_DIR = Path("data/output")
from src.tui.run.helpers import (
    Programa,
    estimar_tiempo,
    formatear_duracion,
    listar_estrategias,
)
from src.tui.test.helpers import listar_patrones


class ProgramCard(Widget):
    """Tarjeta de programa: selectores + botón empezar + barra de progreso.

    Mensajes que burbujean:
      - ProgramCard.Iniciado(nombre)     → on_program_card_iniciado
      - ProgramCard.Eliminado(nombre)    → on_program_card_eliminado
      - ProgramCard.Configurado(nombre, dataset, patron, estrategia)
    """

    class Iniciado(Message):
        def __init__(self, nombre: str) -> None:
            self.nombre = nombre
            super().__init__()

    class Eliminado(Message):
        def __init__(self, nombre: str) -> None:
            self.nombre = nombre
            super().__init__()

    class Cancelado(Message):
        def __init__(self, nombre: str) -> None:
            self.nombre = nombre
            super().__init__()

    class Configurado(Message):
        def __init__(
            self, nombre: str, dataset: str, patron: str, estrategia: str
        ) -> None:
            self.nombre = nombre
            self.dataset = dataset
            self.patron = patron
            self.estrategia = estrategia
            super().__init__()

    def __init__(self, programa: Programa, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__programa = programa
        self._ejecutando = False

    @property
    def nombre_programa(self) -> str:
        return self.__programa.nombre

    @property
    def seleccionado(self) -> bool:
        try:
            return self.query_one(".card-check", Checkbox).value
        except Exception:
            return False

    @property
    def ejecutando(self) -> bool:
        return self._ejecutando

    def marcar_seleccionado(self, valor: bool) -> None:
        try:
            self.query_one(".card-check", Checkbox).value = valor
        except Exception:
            pass

    def compose(self) -> ComposeResult:
        p = self.__programa
        with Horizontal(classes="card-header"):
            yield Checkbox(value=False, classes="card-check")
            yield Label(p.nombre, classes="card-titulo")
            yield Button("✕", variant="error", classes="card-del")

        with Horizontal(classes="card-selectores"):
            redes_disponibles = listar_redes()
            dataset_valido = (
                p.dataset
                if p.dataset and p.dataset in redes_disponibles
                else Select.BLANK
            )
            yield Select(
                options=[(n, n) for n in redes_disponibles],
                prompt="Dataset",
                value=dataset_valido,
                classes="card-select",
                id=f"sel-ds-{p.nombre}",
            )
            patrones_disponibles = listar_patrones()
            patron_valido = (
                p.patron
                if p.patron and p.patron in patrones_disponibles
                else Select.BLANK
            )
            yield Select(
                options=[(n, n) for n in patrones_disponibles],
                prompt="Patrón",
                value=patron_valido,
                classes="card-select",
                id=f"sel-pat-{p.nombre}",
            )
            # Get list of available strategies and verify strategy is valid
            estrategias_disponibles = listar_estrategias()
            estrategia_valida = (
                p.estrategia 
                if p.estrategia and p.estrategia in estrategias_disponibles 
                else Select.BLANK
            )
            yield Select(
                options=[(n, n) for n in estrategias_disponibles],
                prompt="Estrategia",
                value=estrategia_valida,
                classes="card-select",
                id=f"sel-str-{p.nombre}",
            )
            tiene_checkpoint = (_OUTPUT_DIR / f"{p.nombre}.csv").exists()
            yield Button(
                "Continuar" if tiene_checkpoint else "Empezar",
                variant="success",
                classes="btn-empezar",
            )

        yield ProgressBar(total=100, show_eta=False, classes="card-progress")
        yield Label("", classes="card-combo")
        yield Label("--:--:--", classes="card-tiempo")
        yield Rule()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if "card-del" in event.button.classes:
            self.post_message(self.Eliminado(self.__programa.nombre))
        elif "btn-empezar" in event.button.classes:
            if self._ejecutando:
                self.post_message(self.Cancelado(self.__programa.nombre))
            else:
                self.post_message(self.Iniciado(self.__programa.nombre))

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        event.stop()
        self.set_class(event.value, "selected")

    def on_select_changed(self, event: Select.Changed) -> None:
        """Cuando cambia cualquier selector, notificar configuración."""
        event.stop()
        sel_id = event.select.id or ""
        valor = str(event.value) if event.value is not Select.BLANK else ""
        nombre = self.__programa.nombre

        if sel_id.startswith("sel-ds-"):
            self.__programa.dataset = valor
        elif sel_id.startswith("sel-pat-"):
            self.__programa.patron = valor
        elif sel_id.startswith("sel-str-"):
            self.__programa.estrategia = valor

        # Actualizar estimación de tiempo
        self.__actualizar_estimacion()

        self.post_message(
            self.Configurado(
                nombre,
                self.__programa.dataset,
                self.__programa.patron,
                self.__programa.estrategia,
            )
        )

    def __actualizar_estimacion(self) -> None:
        """Actualizar la etiqueta de tiempo estimado basado en el dataset.

        n_dims se lee barato (nombre/primera línea); NO se carga el TPM completo.
        """
        if not self.__programa.dataset:
            return
        try:
            from src.io.manager import dims_de_red

            n_dims = dims_de_red(self.__programa.dataset)
            if n_dims <= 0:
                return
            segs = estimar_tiempo(n_dims)
            lbl = self.query_one(".card-tiempo", Label)
            lbl.update(f"Est: {formatear_duracion(segs)} ({n_dims} dims)")
        except Exception:
            pass

    def marcar_ejecutando(self, en_ejecucion: bool) -> None:
        """Alterna el botón entre Empezar/Continuar y Cancelar."""
        self._ejecutando = en_ejecucion
        btn = self.query_one(".btn-empezar", Button)
        btn.disabled = False
        btn.variant  = "error" if en_ejecucion else "success"
        if en_ejecucion:
            btn.label = "Cancelar"
        else:
            csv_existe = (_OUTPUT_DIR / f"{self.__programa.nombre}.csv").exists()
            btn.label = "Continuar" if csv_existe else "Empezar"

    def marcar_cancelando(self) -> None:
        """Estado de espera: cancelación pedida, worker aún terminando la fila actual."""
        btn = self.query_one(".btn-empezar", Button)
        btn.label    = "Cancelando…"
        btn.variant  = "warning"
        btn.disabled = True

    def actualizar_progreso(self, valor: float) -> None:
        """API pública para actualizar la barra de progreso (0-100)."""
        pb = self.query_one(ProgressBar)
        pb.update(progress=valor)

    def actualizar_combo(self, texto: str) -> None:
        """Muestra el test actual: 'Test N/M: (... | ... | ... | ...)  →  (... | ...)'."""
        self.query_one(".card-combo", Label).update(texto)
