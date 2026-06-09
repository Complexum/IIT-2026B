"""Dataset tab: view, create, and operate on networks in real-time."""

from textual import work
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Label, Static

from src.io.generator import generar_red, peso_estimado
from src.io.manager import (
    cargar_mpt,
    crear_sistema,
    desoptimizar_red,
    dims_de_red,
    eliminar_red,
    listar_redes,
    optimizar_red,
    red_optimizada,
)
from src.tui.dataset.helpers import formatear_sistema, parsear_estado, parsear_indices
from src.tui.dataset.modal import DIMS_WARN_THRESHOLD, ConfirmarCargaModal
from src.tui.dataset.widgets import NetworkItem
from src.tui.shared.consts import estilizar


class DatasetScreen(Widget):
    """Dataset tab: view, create, and operate on networks."""

    DEFAULT_CSS = estilizar()

    selected_network: reactive[str] = reactive("")

    # ── Layout ──

    def compose(self) -> ComposeResult:
        # Row 1: Toolbar — create new networks
        with Horizontal(id="toolbar"):
            yield Input(placeholder="Dims (1-30)", id="inp-dims")
            yield Label("~0 GB", id="lbl-size")
            yield Checkbox("Discretos", id="chk-discretos", value=True)
            # yield Horizontal(
            #     Static("Discretos/Continuos:", classes="label"),
            #     Switch(value=True, id="chk-discretos"),
            #     classes="container",
            # )
            yield Button("+ New", variant="primary", id="btn-new")
            # Optimización por red: genera un sidecar .npy (mmap) sin tocar el CSV.
            yield Checkbox(
                "Optimizar .npy", id="chk-optimizar", value=False, disabled=True
            )

        # Row 2: Two-column bento grid
        with Container(id="main-area"):
            # Left column: scrollable network list
            with VerticalScroll(id="left-pane"):
                yield Label("Networks", classes="pane-title")
                # NetworkItem widgets are added dynamically in on_mount

            # Right column: operation inputs + 3 display columns
            with Vertical(id="right-pane"):
                # 4 text inputs that control the operations below
                with Horizontal(id="inputs-row"):
                    with Vertical(classes="inp-group"):
                        yield Label("Estado inicial", classes="inp-label")
                        yield Input(placeholder="ej: 100", id="inp-estado")
                    with Vertical(classes="inp-group"):
                        yield Label("Condiciones", classes="inp-label")
                        yield Input(placeholder="ej: 111", id="inp-condiciones")
                    with Vertical(classes="inp-group"):
                        yield Label("Alcance", classes="inp-label")
                        yield Input(placeholder="ej: 101", id="inp-alcance")
                    with Vertical(classes="inp-group"):
                        yield Label("Mecanismo", classes="inp-label")
                        yield Input(placeholder="ej: 001", id="inp-mecanismo")

                # 3 scrollable columns: Completo → Candidato → Subsistema
                with Horizontal(id="display-row"):
                    with VerticalScroll(classes="display-col"):
                        yield Label("Sistema Completo", classes="col-title")
                        yield Static("Selecciona una red ←", id="col-completo")
                    with VerticalScroll(classes="display-col"):
                        yield Label("Candidato", classes="col-title")
                        yield Static("", id="col-candidato")
                    with VerticalScroll(classes="display-col"):
                        yield Label("Subsistema", classes="col-title")
                        yield Static("", id="col-subsistema")

    def on_mount(self) -> None:
        """Load the network list from disk when the tab is first shown."""
        # Caché de TPMs cargadas: evita releer el CSV completo en cada tecleo.
        self.__tpm_cache: dict[str, "np.ndarray"] = {}
        # Evita que el set programático del checkbox dispare optimizar/desoptimizar.
        self.__sync_optimizar = False
        self.__refresh_network_list()

    def __sync_checkbox_optimizar(self, name: str) -> None:
        """Refleja en el checkbox si la red seleccionada tiene sidecar .npy."""
        chk = self.query_one("#chk-optimizar", Checkbox)
        self.__sync_optimizar = True
        try:
            chk.disabled = not name
            chk.value = bool(name) and red_optimizada(name)
        finally:
            self.__sync_optimizar = False

    def __tpm(self, name: str) -> "np.ndarray":
        """TPM de una red, cargada una sola vez y reusada."""
        tpm = self.__tpm_cache.get(name)
        if tpm is None:
            tpm = cargar_mpt(name)
            self.__tpm_cache[name] = tpm
        return tpm

    # ── Reactive watcher ──

    def watch_selected_network(self, old_value: str, new_value: str) -> None:
        """Visually highlight the selected network in the left pane."""
        for item in self.query(NetworkItem):
            item.add_class(
                "selected"
            ) if item.network_name == new_value else item.remove_class("selected")

    # ── Event handlers ──

    # Textual resolves handler names from the message class path:
    #   Input.Changed        → on_input_changed(event)
    #   Button.Pressed       → on_button_pressed(event)
    #   NetworkItem.Selected → on_network_item_selected(event)
    #   NetworkItem.Deleted  → on_network_item_deleted(event)

    def on_input_changed(self, event: Input.Changed) -> None:
        """React to any Input change in real-time.

        - inp-dims:        update the estimated size label
        - inp-estado/cond/alc/mec: recalculate the 3 display columns
        """
        match event.input.id:
            case "inp-dims":
                self.__update_size_estimate(event.value)
            case "inp-estado" | "inp-condiciones" | "inp-alcance" | "inp-mecanismo":
                self.__update_display()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle toolbar buttons (NetworkItem buttons are stopped internally)."""
        if event.button.id == "btn-new":
            self.__create_network()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        """Toggle de optimización (.npy) para la red seleccionada."""
        if event.checkbox.id != "chk-optimizar" or self.__sync_optimizar:
            return
        name = self.selected_network
        if not name:
            return
        event.checkbox.disabled = True  # bloquear hasta terminar
        self.__toggle_optimizacion(name, event.value)

    @work(thread=True, exclusive=True)
    def __toggle_optimizacion(self, name: str, activar: bool) -> None:
        """Genera/elimina el sidecar .npy en background (puede tardar en redes grandes)."""
        try:
            if activar:
                optimizar_red(name)
                msg = f"'{name}' optimizada (.npy)"
            else:
                desoptimizar_red(name)
                msg = f"'{name}' sin optimizar (CSV)"
            # Forzar relectura: la próxima carga usará mmap del .npy (o el CSV).
            self.__tpm_cache.pop(name, None)
        except Exception as e:  # noqa: BLE001
            msg = f"Error optimizando '{name}': {e}"

        def _finish() -> None:
            if self.selected_network == name:
                self.__sync_checkbox_optimizar(name)
            self.notify(msg)

        self.app.call_from_thread(_finish)

    def on_network_item_selected(self, event: NetworkItem.Selected) -> None:
        """A network name was clicked — warn if large, then load."""
        n_dims = dims_de_red(event.name)
        if n_dims >= DIMS_WARN_THRESHOLD:

            def _on_confirm(ok: bool) -> None:
                if ok:
                    self.selected_network = event.name
                    self.__load_and_display(event.name)

            self.app.push_screen(ConfirmarCargaModal(event.name, n_dims), _on_confirm)
        else:
            self.selected_network = event.name
            self.__load_and_display(event.name)

    def on_network_item_deleted(self, event: NetworkItem.Deleted) -> None:
        """Delete button (✕) clicked — remove the file and refresh the list."""
        eliminar_red(event.name)
        self.__tpm_cache.pop(event.name, None)
        if self.selected_network == event.name:
            self.selected_network = ""
            self.__clear_display()
        self.__refresh_network_list()
        self.__notify_execution_refresh()

    def __notify_execution_refresh(self) -> None:
        from src.tui.run.screen import ExecutionScreen

        try:
            self.app.query_one(ExecutionScreen).refrescar_datasets()
        except Exception:
            pass

    # ── Helpers Privados ──

    def __update_size_estimate(self, value: str) -> None:
        """Show the estimated file size for a network of N dimensions."""
        lbl = self.query_one("#lbl-size", Label)
        try:
            dims = int(value)
            if dims < 1:
                raise ValueError
            gb = peso_estimado(dims)
            lbl.update(f"~{gb * 1024:.2f} MB" if gb < 0.001 else f"~{gb:.3f} GB")
        except (ValueError, TypeError):
            lbl.update("~0 GB")

    def __create_network(self) -> None:
        """Create a new network CSV from the toolbar inputs."""
        inp = self.query_one("#inp-dims", Input)
        chk = self.query_one("#chk-discretos", Checkbox)
        try:
            dims = int(inp.value)
            if not (1 <= dims <= 30):
                return
        except (ValueError, TypeError):
            return
        # generar_red siempre crea un nombre nuevo (no sobreescribe), así que
        # no hay TPM cacheada que invalidar aquí.
        generar_red(dims, datos_deterministas=chk.value)
        self.__refresh_network_list()
        self.__notify_execution_refresh()

    def __refresh_network_list(self) -> None:
        """Reload the network list from data/input/networks/."""
        pane = self.query_one("#left-pane", VerticalScroll)
        # Remove only NetworkItem widgets (keep the "Networks" title Label)
        for item in list(pane.query(NetworkItem)):
            item.remove()
        # Mount fresh items
        for name in listar_redes():
            pane.mount(NetworkItem(name))

    def __load_and_display(self, name: str) -> None:
        """Load a network and set sensible default input values."""
        # Reflejar el estado de optimización de la red recién seleccionada.
        self.__sync_checkbox_optimizar(name)
        try:
            tpm = self.__tpm(name)
            n_dims = tpm.shape[1]

            # Reset operation inputs for the new network
            self.query_one("#inp-condiciones", Input).value = "1" * n_dims
            self.query_one("#inp-alcance", Input).value = "1" * n_dims
            self.query_one("#inp-mecanismo", Input).value = "1" * n_dims
            self.query_one("#inp-estado", Input).value = f"1{'0' * (n_dims - 1)}"
            # Explicit call: Input.Changed may not fire if the string is unchanged
            # (e.g. two networks with the same number of dimensions).
            self.__update_display()

        except Exception as e:
            self.query_one("#col-completo", Static).update(f"Error: {e}")

    def __update_display(self) -> None:
        """Recalculate and render the 3 display columns.

        Pipeline:
          TPM + estado → Sistema Completo
          + condiciones → Candidato (condicionado)
          + alcance/mecanismo → Subsistema (substraído)
        """
        if not self.selected_network:
            return

        try:
            tpm = self.__tpm(self.selected_network)
            n_dims = tpm.shape[1]

            # ── Parse estado ──
            estado_str = self.query_one("#inp-estado", Input).value
            estado = parsear_estado(estado_str)
            if estado is None:
                estado = tuple(0 for _ in range(n_dims))
            # Pad or truncate to match network dimensions
            if len(estado) < n_dims:
                estado = estado + (0,) * (n_dims - len(estado))
            elif len(estado) > n_dims:
                estado = estado[:n_dims]

            # ── Column 1: Sistema Completo ──
            sistema = crear_sistema(tpm, estado)
            self.query_one("#col-completo", Static).update(formatear_sistema(sistema))

            # ── Column 2: Candidato (condicionado) ──
            cond_str = self.query_one("#inp-condiciones", Input).value
            condiciones = parsear_indices(cond_str)
            candidato = sistema.condicionar(condiciones) if condiciones else sistema
            self.query_one("#col-candidato", Static).update(
                formatear_sistema(candidato)
            )

            # ── Column 3: Subsistema (substraído) ──
            alc_str = self.query_one("#inp-alcance", Input).value
            mec_str = self.query_one("#inp-mecanismo", Input).value
            alcance = parsear_indices(alc_str)
            mecanismo = parsear_indices(mec_str)

            if alcance or mecanismo:
                subsistema = candidato.substraer(alcance, mecanismo)
            else:
                subsistema = candidato
            self.query_one("#col-subsistema", Static).update(
                formatear_sistema(subsistema)
            )

        except Exception as e:
            self.query_one("#col-completo", Static).update(f"Error: {e}")

    def __clear_display(self) -> None:
        """Clear all display columns and operation inputs."""
        self.__sync_checkbox_optimizar("")  # sin red → checkbox off/disabled
        for col_id in ("#col-completo", "#col-candidato", "#col-subsistema"):
            self.query_one(col_id, Static).update("")
        for inp_id in (
            "#inp-estado",
            "#inp-condiciones",
            "#inp-alcance",
            "#inp-mecanismo",
        ):
            self.query_one(inp_id, Input).value = ""
