"""Comando: edit execution <nombre> [--dataset ...] [--patron ...] [--estrategia ...]."""

from src.cli.utils import error, info, parse_kv, success, warn
from src.tui.run.helpers import (
    cargar_programa,
    guardar_programa,
    listar_programas,
    podar_opciones,
)


def handle(args) -> None:
    if args.resource != "execution":
        error("Solo se puede editar 'execution'.")
        return

    name = args.name
    if name not in listar_programas():
        error(f"Execution '{name}' no encontrado.")
        return

    prog = cargar_programa(name)
    changed = False

    if args.dataset is not None:
        prog.dataset = args.dataset
        changed = True
    if args.patron is not None:
        prog.patron = args.patron
        changed = True
    if args.estrategia is not None:
        if prog.estrategia != args.estrategia and prog.opciones:
            # Las opciones son por estrategia (`SIA.opciones`); las de la anterior
            # no aplican y harían fallar la nueva al ejecutar.
            prog.opciones, descartadas = podar_opciones(args.estrategia, prog.opciones)
            if descartadas:
                warn(
                    f"Descartadas opciones que '{args.estrategia}' no admite: "
                    + ", ".join(f"{k}={v}" for k, v in sorted(descartadas.items()))
                )
        prog.estrategia = args.estrategia
        changed = True
    if args.opcion:
        try:
            nuevas = parse_kv(args.opcion)
        except ValueError as e:
            error(str(e))
            return
        # 'attr=' borra la opción; el resto se acumula sobre las existentes.
        opciones = dict(prog.opciones or {})
        for k, v in nuevas.items():
            opciones.pop(k, None) if v == "" else opciones.update({k: v})
        prog.opciones = opciones
        info(f"Opciones: {opciones or '(ninguna)'}")
        changed = True

    if changed:
        guardar_programa(prog)
        success(f"Execution '{name}' actualizado.")
    else:
        error(
            "No se proporcionaron campos para editar. Usa --dataset, --patron, --estrategia o --opcion."
        )
