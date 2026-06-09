"""Widgets del tab Analysis: ResultCheckbox con texto wrappeable."""

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


class ResultCheckbox(Widget):
    """Checkbox custom con Static (wrap) en vez de Checkbox (truncate).

    Layout: [✓] | nombre largo que hace wrap
    """

    value: reactive[bool] = reactive(False)

    class Changed(Message):
        def __init__(self, id_str: str, value: bool) -> None:
            self.id_str = id_str
            self.value = value
            super().__init__()

    def __init__(self, label: str, id_str: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._label_text = label
        self._id_str = id_str

    def compose(self) -> ComposeResult:
        with Horizontal(classes="cb-row"):
            yield Static(self._render_indicator(), id="cb-indicator", classes="cb-box")
            yield Static(self._label_text, id="cb-text", classes="cb-label")

    def _render_indicator(self) -> str:
        return "[✓]" if self.value else "[ ]"

    def on_click(self) -> None:
        """Toggle value on click."""
        self.value = not self.value
        self.query_one("#cb-indicator", Static).update(self._render_indicator())
        self.post_message(self.Changed(self._id_str, self.value))

    def watch_value(self, old: bool, new: bool) -> None:
        if old != new:
            try:
                self.query_one("#cb-indicator", Static).update(self._render_indicator())
            except Exception:
                pass
