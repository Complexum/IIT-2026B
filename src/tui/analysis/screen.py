"""Compare tab: comparar N resultados agrupados por dataset+patron."""

from pathlib import Path

import polars as pl
from textual.app import ComposeResult
from textual.containers import Container, ScrollableContainer, VerticalScroll
from textual.widget import Widget
from textual.widgets import Button, Input, Label, Select, Static

from src.io.manager import listar_resultados
from src.tui.run.helpers import cargar_programa, extract_program_name
from src.tui.shared.consts import estilizar

from .compare import (
    build_rich_table,
    build_rich_table_n,
    compare_group_n,
    compare_results,
    group_selected,
    parse_result_key,
)
from .plots import generate_analysis_page, generate_comparison_plot, open_plot
from .widgets import ResultCheckbox

_TOL_DEFAULT = 1e-4
_OUTPUT_DIR = Path("data/output")
_PICS_DIR = Path(".docs/papers/project/pics")


class AnalysisScreen(Widget):
    """Compare tab: comparar pérdida entre resultados CSV agrupados por dataset+patron."""

    DEFAULT_CSS = estilizar()

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._df: pl.DataFrame | None = None
        self._label_a = ""
        self._label_b = ""
        self._tol = _TOL_DEFAULT
        self._groups: dict[tuple[str, str], list[str]] = {}
        self._all_strategies: list[str] = []
        self._select_all_state = False

    def compose(self) -> ComposeResult:
        with Container(id="cmp-grid"):
            with VerticalScroll(id="cmp-left"):
                yield Label("Resultados", classes="pane-title")
            with Container(id="cmp-right"):
                with Container(id="cmp-controls"):
                    yield Button("Sel. todo", id="cmp-select-all", variant="default")
                    yield Label("tol:", id="cmp-tol-label")
                    yield Input(value="0.0001", id="cmp-tol-input")
                    yield Label("Ref:", id="cmp-ref-label")
                    yield Select([], id="cmp-ref-select", prompt="auto")
                    yield Button("Comparar", id="cmp-btn", variant="primary")
                    yield Button(
                        "Graficar", id="cmp-plot-btn", variant="success", disabled=True
                    )
                with ScrollableContainer(id="cmp-output"):
                    yield Label(
                        "Selecciona 2+ resultados ←",
                        id="cmp-title",
                        classes="content-title",
                    )
                    yield Static("", id="cmp-summary")
                    yield Static("", id="cmp-table")

    def on_mount(self) -> None:
        self._refresh_list()

    def refrescar(self) -> None:
        self._refresh_list()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cmp-btn":
            self._run_compare()
        elif event.button.id == "cmp-plot-btn":
            self._open_plots()
        elif event.button.id == "cmp-select-all":
            self._toggle_select_all()

    def _sanitize_id(self, nombre: str) -> str:
        return nombre.replace("/", "_").replace("--", "_").replace("-", "_")

    def _refresh_list(self) -> None:
        pane = self.query_one("#cmp-left", VerticalScroll)
        for cb in list(pane.query(ResultCheckbox)):
            cb.remove()
        self._check_states: dict[str, bool] = {}
        self._id_mapping: dict[str, str] = {}
        self._all_strategies = []

        for idx, nombre in enumerate(listar_resultados()):
            strat = ""
            try:
                prog_name = extract_program_name(nombre)
                strat = cargar_programa(prog_name).estrategia
            except Exception:
                pass
            if strat and strat not in self._all_strategies:
                self._all_strategies.append(strat)
            etiqueta = nombre + (f" [{strat}]" if strat else "")
            safe_id = f"cb_{idx}"
            self._id_mapping[safe_id] = nombre
            self._check_states[safe_id] = False
            pane.mount(ResultCheckbox(etiqueta, safe_id))

        # Poblar el selector de referencia
        ref_select = self.query_one("#cmp-ref-select", Select)
        options = [(s, s) for s in sorted(self._all_strategies)]
        ref_select.set_options(options)
        if "phi" in self._all_strategies:
            ref_select.value = "phi"
        elif self._all_strategies:
            ref_select.value = sorted(self._all_strategies)[0]

    def _toggle_select_all(self) -> None:
        self._select_all_state = not self._select_all_state
        for safe_id in self._check_states:
            self._check_states[safe_id] = self._select_all_state
        for cb in self.query(ResultCheckbox):
            cb.value = self._select_all_state
            if self._select_all_state:
                cb.add_class("selected")
            else:
                cb.remove_class("selected")

    def on_result_checkbox_changed(self, event: ResultCheckbox.Changed) -> None:
        self._check_states[event.id_str] = event.value
        for cb in self.query(ResultCheckbox):
            if cb._id_str == event.id_str:
                if event.value:
                    cb.add_class("selected")
                else:
                    cb.remove_class("selected")

    def _selected(self) -> list[str]:
        return [
            self._id_mapping[id_str]
            for id_str, value in self._check_states.items()
            if value and id_str in self._id_mapping
        ]

    def _get_tol(self) -> float:
        try:
            return float(self.query_one("#cmp-tol-input", Input).value)
        except ValueError:
            return _TOL_DEFAULT

    def _get_reference(self) -> str | None:
        try:
            val = self.query_one("#cmp-ref-select", Select).value
            return str(val) if val and val is not Select.BLANK else None
        except Exception:
            return None

    def _run_compare(self) -> None:
        sel = self._selected()
        title = self.query_one("#cmp-title", Label)
        summary_w = self.query_one("#cmp-summary", Static)
        table_w = self.query_one("#cmp-table", Static)
        plot_btn = self.query_one("#cmp-plot-btn", Button)

        plot_btn.disabled = True
        self._df = None
        self._groups = {}

        if len(sel) < 2:
            title.update(f"Selecciona 2+ resultados ({len(sel)} seleccionados)")
            summary_w.update("")
            table_w.update("")
            return

        tol = self._get_tol()
        self._tol = tol
        self._groups = group_selected(sel)

        n_groups = len(self._groups)
        n_strats_total = sum(len(v) for v in self._groups.values())
        title.update(
            f"{n_strats_total} resultados · {n_groups} grupo(s) · tol={tol:.0e}"
        )

        # Comparar cada grupo
        lines: list[str] = []
        all_tables = []
        for (dataset, patron), names in self._groups.items():
            lines.append(f"[bold cyan]{dataset} | {patron}[/bold cyan]")
            if len(names) < 2:
                strat = parse_result_key(names[0])[1]
                lines.append(f"  Solo 1 estrategia: {strat} (nada que comparar)")
                continue
            try:
                res = compare_group_n(names, tol)
                strats = res["strategies"]
                lines.append(f"  Estrategias: {', '.join(strats)}")
                for (a, b), s in res["pairs"].items():
                    pct = 100 * s["n_ok"] / s["n"] if s["n"] > 0 else 0
                    lines.append(
                        f"  {a} vs {b}: {s['n_ok']}/{s['n']} ({pct:.1f}%) "
                        f"max={s['max_d']:.6f}"
                    )
                all_tables.append(build_rich_table_n(res["pairs"], strats, tol))
            except Exception as exc:
                lines.append(f"  [red]Error: {exc}[/red]")

        summary_w.update("\n".join(lines))
        if all_tables:
            table_w.update(all_tables[0])
        plot_btn.disabled = False

    def _open_plots(self) -> None:
        if not self._groups:
            return

        title = self.query_one("#cmp-title", Label)
        reference = self._get_reference()
        tol = self._tol
        sel = self._selected()

        # Plotly interactivo
        try:
            out_path = generate_analysis_page(self._groups, reference, tol)
            open_plot(out_path)
        except ImportError:
            title.update("plotly no disponible — instalar con: uv add plotly")
            return
        except Exception as exc:
            title.update(f"Error plotly: {exc}")
            return

        # Paper figures (matplotlib)
        paper_paths: list[Path] = []
        try:
            from src.paper.plots import generate_paper_figures

            paper_paths = generate_paper_figures(sel, _OUTPUT_DIR, _PICS_DIR, reference)
        except Exception as exc:
            title.update(f"Plotly OK · Error paper: {exc}")
            return

        paper_names = ", ".join(p.name for p in paper_paths)
        title.update(f"Plotly → {out_path.name} · Paper → {paper_names} ({_PICS_DIR})")
