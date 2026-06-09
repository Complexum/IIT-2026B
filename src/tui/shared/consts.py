"""Utilidades compartidas para la TUI."""

import inspect
from pathlib import Path


def estilizar(archivo: str = "styles.scss") -> str:
    """Carga CSS adyacente al archivo que llama esta función.

    Busca `archivo` en el mismo directorio que el fichero invocador.
    Esto permite que cada módulo tenga su propio styles.scss junto al .py.
    """
    fn = inspect.stack()[1].filename
    ruta = Path(fn).parent / archivo
    return ruta.read_text(encoding="utf-8") if ruta.exists() else ""


# Alias
estilos = estilizar
