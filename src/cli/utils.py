"""Helpers compartidos del CLI."""

from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def error(msg: str) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {msg}")


def success(msg: str) -> None:
    console.print(f"[bold green]✓[/bold green] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]⚠[/bold yellow] {msg}")


def info(msg: str) -> None:
    console.print(msg)


def print_table(columns: list[str], rows: list[list[Any]], title: str = "") -> None:
    table = Table(title=title)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*[str(v) for v in row])
    console.print(table)


def parse_kv(args: list[str]) -> dict[str, str]:
    """Parsea argumentos tipo 'clave=valor' en un dict."""
    result: dict[str, str] = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"Formato inválido: {arg!r} (esperado clave=valor)")
        k, v = arg.split("=", 1)
        result[k.strip()] = v.strip()
    return result
