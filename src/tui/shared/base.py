from textual.reactive import Reactive
from textual.widgets import Footer, Header


class IITHeader(Header):
    icon: Reactive[str] = Reactive(">")


class IITFooter(Footer):
    pass
