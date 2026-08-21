"""Comando: new dataset|execution|patron <nombre|dims> [flags]."""

from src.cli.utils import error, success, warn
from src.io.generator import generar_red
from src.io.manager import listar_redes
from src.tui.run.helpers import (
    Programa,
    guardar_programa,
    listar_estrategias,
    listar_programas,
    siguiente_nombre_programa,
)
from src.tui.test.helpers import (
    Patron,
    guardar_patron,
    listar_patrones,
    siguiente_nombre_patron,
)


def handle(args) -> None:
    resource = args.resource

    if resource == "dataset":
        try:
            dims = int(args.name_or_dims)
        except ValueError:
            error("Las dimensiones deben ser un número entero.")
            return
        if not (1 <= dims <= 30):
            error("Las dimensiones deben estar entre 1 y 30.")
            return
        nombre = generar_red(dims, datos_deterministas=args.discretos)
        success(
            f"Dataset '{nombre}' creado ({dims} dims, {'determinista' if args.discretos else 'probabilista'})."
        )

    elif resource == "execution":
        name = args.name_or_dims
        if not name:
            name = siguiente_nombre_programa()
        if name in listar_programas():
            error(f"Execution '{name}' ya existe.")
            return

        prog = Programa(nombre=name)
        if args.dataset:
            if args.dataset not in listar_redes():
                warn(f"Dataset '{args.dataset}' no existe aún.")
            prog.dataset = args.dataset
        if args.patron:
            prog.patron = args.patron
        if args.estrategia:
            if args.estrategia not in listar_estrategias():
                warn(f"Estrategia '{args.estrategia}' no reconocida.")
            prog.estrategia = args.estrategia
        if getattr(args, "opcion", None):
            from src.cli.utils import parse_kv

            try:
                prog.opciones = parse_kv(args.opcion)
            except ValueError as e:
                error(str(e))
                return

        guardar_programa(prog)
        success(f"Execution '{name}' creado.")

    elif resource == "patron":
        name = args.name_or_dims
        if not name:
            name = siguiente_nombre_patron()
        if name in listar_patrones():
            error(f"Patrón '{name}' ya existe.")
            return
        patron = Patron(nombre=name)
        guardar_patron(patron)
        success(
            f"Patrón '{name}' creado (vacío). Edítalo con el TUI o directamente en el JSON."
        )

    else:
        warn(f"Recurso desconocido: {resource}")
