"""Comando: delete dataset|execution|patron <nombre>."""

from src.cli.utils import error, success, warn
from src.io.manager import eliminar_red, listar_redes
from src.tui.run.helpers import eliminar_programa, listar_programas
from src.tui.test.helpers import eliminar_patron, listar_patrones


def handle(args) -> None:
    resource = args.resource
    name = args.name

    if resource == "dataset":
        if name not in listar_redes():
            error(f"Dataset '{name}' no encontrado.")
            return
        eliminar_red(name)
        success(f"Dataset '{name}' eliminado.")

    elif resource == "execution":
        if name not in listar_programas():
            error(f"Execution '{name}' no encontrado.")
            return
        eliminar_programa(name)
        success(f"Execution '{name}' eliminado.")

    elif resource == "patron":
        if name not in listar_patrones():
            error(f"Patrón '{name}' no encontrado.")
            return
        eliminar_patron(name)
        success(f"Patrón '{name}' eliminado.")

    else:
        warn(f"Recurso desconocido: {resource}")
