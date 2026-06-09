"""Aplicación TUI principal — punto de entrada para la interfaz Textual."""

import sys

from textual.app import App, ComposeResult
from textual.widgets import TabbedContent, TabPane

from src.tui.analysis.screen import AnalysisScreen
from src.tui.dataset.screen import DatasetScreen
from src.tui.results.screen import ResultsScreen
from src.tui.run.screen import ExecutionScreen
from src.tui.shared.base import IITFooter, IITHeader
from src.tui.shared.consts import estilizar
from src.tui.test.screen import TestingScreen


class MIPApp(App):
    """MIP-IIT Solver TUI Application."""

    TITLE = "IIT-MIP"
    SUB_TITLE = "Integrated Information Theory"

    CSS = estilizar()

    BINDINGS = [
        ("d", "show_tab('dataset')", "Dataset"),
        ("t", "show_tab('testing')", "Testing"),
        ("e", "show_tab('execution')", "Execution"),
        ("r", "show_tab('results')", "Results"),
        ("a", "show_tab('analysis')", "Analysis"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield IITHeader()
        with TabbedContent(initial="dataset"):
            with TabPane("Dataset", id="dataset"):
                yield DatasetScreen()
            with TabPane("Testing", id="testing"):
                yield TestingScreen()
            with TabPane("Execution", id="execution"):
                yield ExecutionScreen()
            with TabPane("Results", id="results"):
                yield ResultsScreen()
            with TabPane("Analysis", id="analysis"):
                yield AnalysisScreen()
        yield IITFooter()

    def action_show_tab(self, tab: str) -> None:
        self.query_one(TabbedContent).active = tab


def _run_app():
    """Inicia la aplicación TUI."""
    app = MIPApp()
    app.run()


def main():
    no_watch = "--no-watch" in sys.argv

    if no_watch:
        _run_app()
    else:
        # Auto-reload: observa cambios en .py y reinicia el proceso.
        from src.tui.reloader import run_with_reloader

        run_with_reloader(_run_app)


if __name__ == "__main__":
    main()
