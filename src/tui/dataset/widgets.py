from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button


class NetworkItem(Widget):
    """A single network entry: clickable name + delete (✕) button.

    Posts two custom messages that bubble up to the parent:
      - NetworkItem.Selected(name)  → when the name button is clicked
      - NetworkItem.Deleted(name)   → when the ✕ button is clicked

    Usage in parent handler:
        def on_network_item_selected(self, event: NetworkItem.Selected):
            print(event.name)
    """

    class Selected(Message):
        """Posted when the network name is clicked."""

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    class Deleted(Message):
        """Posted when the ✕ button is clicked."""

        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    def __init__(self, name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.network_name = name

    def compose(self) -> ComposeResult:
        yield Button(self.network_name, classes="net-name")
        yield Button("✕", variant="error", classes="net-del")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route internal button clicks to custom messages.

        event.stop() prevents the generic Button.Pressed from bubbling
        up — only our Selected/Deleted messages reach the parent.
        """
        event.stop()
        if "net-del" in event.button.classes:
            self.post_message(self.Deleted(self.network_name))
        else:
            self.post_message(self.Selected(self.network_name))
