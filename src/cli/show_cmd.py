"""Comando: show dataset|execution|patron|strategy <nombre>."""

from src.cli.utils import error, info, warn
from src.iit.strategies.python.sia import SIA
from src.io.manager import cargar_mpt, listar_redes
from src.tui.run.helpers import cargar_programa
from src.tui.test.helpers import cargar_patron, listar_patrones


def handle(args) -> None:
    resource = args.resource
    name = args.name

    if resource == "dataset":
        if name not in listar_redes():
            error(f"Dataset '{name}' no encontrado.")
            return
        tpm = cargar_mpt(name)
        info(f"Dataset: [bold]{name}[/bold]")
        info(f"  Dimensiones: {tpm.shape[1]}")
        info(f"  Estados: {tpm.shape[0]}")

    elif resource == "execution":
        try:
            prog = cargar_programa(name)
        except FileNotFoundError:
            error(f"Execution '{name}' no encontrado.")
            return
        info(f"Execution: [bold]{prog.nombre}[/bold]")
        info(f"  Dataset:    {prog.dataset or '—'}")
        info(f"  Patrón:     {prog.patron or '—'}")
        info(f"  Estrategia: {prog.estrategia or '—'}")
        info(f"  Estado:     {prog.estado}")
        info(f"  Progreso:   {prog.progreso:.1f}%")

    elif resource == "patron":
        if name not in listar_patrones():
            error(f"Patrón '{name}' no encontrado.")
            return
        patron = cargar_patron(name)
        info(f"Patrón: [bold]{patron.nombre}[/bold]")
        info(f"  Estados:     {patron.estados}")
        info(f"  Condiciones: {patron.condiciones}")
        info(f"  Alcances:    {patron.alcances}")
        info(f"  Mecanismos:  {patron.mecanismos}")

    elif resource == "strategy":
        from src.iit.strategies.runner import importar_estrategias

        importar_estrategias()
        if name not in SIA.registry:
            error(f"Estrategia '{name}' no encontrada.")
            return
        info(f"Estrategia: [bold]{name}[/bold]")

    else:
        warn(f"Recurso desconocido: {resource}")
