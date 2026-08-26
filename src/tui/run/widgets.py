"""Widgets del tab Execution: ProgramCard."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Label, ProgressBar, Rule, Select

from pathlib import Path


_OUTPUT_DIR = Path("data/output")
from src.tui.run.helpers import (
    Programa,
    estimar_tiempo,
    formatear_duracion,
    listar_estrategias,
    opciones_de_estrategia,
    podar_opciones,
    redes_compatibles,
)
from src.tui.test.helpers import listar_patrones


class OpcionesRow(Widget):
    """Selectores de las opciones de la estrategia (`SIA.opciones`).

    Es un widget aparte —y no parte del `compose` de `ProgramCard`— para poder
    recomponerse solo cuando cambia la estrategia, sin resetear la barra de progreso
    ni el estado de ejecución de la tarjeta.

    No dibuja nada si la estrategia no declara opciones, que es el caso de la
    mayoría (`analytic`, `queyranne`, `phi`).
    """

    def __init__(self, programa: Programa, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__programa = programa

    def compose(self) -> ComposeResult:
        p = self.__programa
        opciones = opciones_de_estrategia(p.estrategia)
        # `Widget.render()` por defecto dibuja el identificador CSS del widget, así
        # que una fila sin hijos mostraba literalmente "OpcionesRow.opciones-row" y
        # además ocupaba alto. Con `display = False` desaparece del layout.
        self.display = bool(opciones)
        if not opciones:
            return
        with Horizontal(classes="card-opciones"):
            for attr, valores in opciones.items():
                yield Label(f"{attr}:", classes="opt-label")
                # El primer valor de cada tupla es el default (`SIA.defaults`).
                actual = p.opciones.get(attr, valores[0])
                if actual not in valores:
                    actual = valores[0]
                yield Select(
                    options=[(v, v) for v in valores],
                    value=actual,
                    allow_blank=False,
                    name=attr,  # el handler lee el atributo de acá
                    classes="card-select-opt",
                )


class ProgramCard(Widget):
    """Tarjeta de programa: selectores + botón empezar + barra de progreso.

    Mensajes que burbujean:
      - ProgramCard.Iniciado(nombre)     → on_program_card_iniciado
      - ProgramCard.Eliminado(nombre)    → on_program_card_eliminado
      - ProgramCard.Configurado(nombre, dataset, patron, estrategia, opciones)
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
            self,
            nombre: str,
            dataset: str,
            patron: str,
            estrategia: str,
            opciones: dict[str, str] | None = None,
        ) -> None:
            self.nombre = nombre
            self.dataset = dataset
            self.patron = patron
            self.estrategia = estrategia
            self.opciones = dict(opciones or {})
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
            # La ESTRATEGIA va primero porque decide el resto: qué redes son
            # válidas (`necesita_espec` / `necesita_mpt`) y qué opciones existen.
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
            redes = redes_compatibles(p.estrategia)
            nombres = {v for _, v in redes}
            dataset_valido = (
                p.dataset if p.dataset and p.dataset in nombres else Select.BLANK
            )
            yield Select(
                options=redes,
                prompt="Red",
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
            tiene_checkpoint = (_OUTPUT_DIR / f"{p.nombre}.csv").exists()
            yield Button(
                "Continuar" if tiene_checkpoint else "Empezar",
                variant="success",
                classes="btn-empezar",
            )

        yield OpcionesRow(p, classes="opciones-row")

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
            # Las opciones son POR estrategia: al cambiarla, las de la anterior
            # dejan de aplicar. Se podan acá también —además de en la pantalla—
            # para que lo que se ve coincida con lo que se guarda.
            self.__programa.opciones, _ = podar_opciones(valor, self.__programa.opciones)
            try:
                self.query_one(OpcionesRow).refresh(recompose=True)
            except Exception:
                pass
            self.__refiltrar_redes()
        elif "card-select-opt" in event.select.classes:
            attr = event.select.name or ""
            if attr:
                self.__programa.opciones[attr] = valor

        # Actualizar estimación de tiempo
        self.__actualizar_estimacion()

        self.post_message(
            self.Configurado(
                nombre,
                self.__programa.dataset,
                self.__programa.patron,
                self.__programa.estrategia,
                self.__programa.opciones,
            )
        )

    def __refiltrar_redes(self) -> None:
        """Deja en el selector sólo las redes que la estrategia puede consumir.

        Se usa `set_options` en vez de recomponer para no perder el resto del
        estado de la tarjeta. Si la red elegida deja de ser válida —por ejemplo al
        pasar a una estrategia que exige especificación— se limpia en vez de
        quedar seleccionada y fallar recién al ejecutar.
        """
        p = self.__programa
        try:
            sel = self.query_one(f"#sel-ds-{p.nombre}", Select)
        except Exception:
            return
        redes = redes_compatibles(p.estrategia)
        sel.set_options(redes)
        if p.dataset and p.dataset in {v for _, v in redes}:
            sel.value = p.dataset
        else:
            p.dataset = ""

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
