"""Barra de identificabilidad: ¿esta red admite una spec compacta encontrable?

Extraída de `DatasetScreen.__veredicto` para no mezclar el diagnóstico con el
formato de la columna "Sistema Completo". Muestra el informe completo de
`io/identificabilidad` (influencias, piso de ruido, fan-in) encima de la fila
de inputs, no dentro de una columna scrolleable.
"""

import numpy as np
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Label, Static

from src.io.identificabilidad import diagnosticar, informe
from src.io.manager import capacidades
from src.tui.shared.consts import estilizar


class VeredictoBar(Widget):
    """Informe de identificabilidad de la red seleccionada."""

    DEFAULT_CSS = estilizar()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        #: `diagnosticar` es O(N·D·2^D): el informe se calcula una vez por red
        #: y no en cada repintado.
        self.__cache: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Label("Identificabilidad", classes="col-title")
        yield Static("—", id="veredicto-text")

    def actualizar(self, nombre: str, tpm: np.ndarray | None = None) -> None:
        """Refrescar la barra para `nombre`. `tpm` puede faltar si sólo hay spec."""
        self.query_one("#veredicto-text", Static).update(self.__informe(nombre, tpm))

    def limpiar(self) -> None:
        self.query_one("#veredicto-text", Static).update("—")

    def __informe(self, nombre: str, tpm: np.ndarray | None) -> str:
        """¿Esta red sirve para la ruta que lee la especificación?

        Si ya tiene `.spec.json` la respuesta es sí y no hay nada que diagnosticar.
        Si no, `identificabilidad` dice si una spec compacta es siquiera
        encontrable — puede no serlo por ser **densa** (incompresible por conteo) o
        porque la señal está **bajo el ruido** de muestreo de la propia TPM.
        """
        if capacidades(nombre)["spec"]:
            return "[b green]Ⓢ tiene especificación — diagnosticar no hace falta[/b green]"
        cacheado = self.__cache.get(nombre)
        if cacheado is None:
            if tpm is None:
                return "—"
            try:
                d = diagnosticar(tpm, nombre)
                cacheado = self.__colorear(informe(d), d["veredicto"])
            except Exception as e:  # noqa: BLE001
                cacheado = f"[b yellow]diagnóstico no disponible ({type(e).__name__})[/b yellow]"
            self.__cache[nombre] = cacheado
        return cacheado

    @staticmethod
    def __colorear(texto: str, veredicto: str) -> str:
        """Colorea la línea final `-> veredicto` del informe."""
        color = "green" if veredicto.startswith("IDENTIFICABLE") else "yellow"
        lineas = texto.splitlines()
        lineas[-1] = f"[b {color}]{lineas[-1]}[/b {color}]"
        return "\n".join(lineas)
