"""Comando: run execution <nombre> [--no-resume]."""

import csv
import time
from pathlib import Path

from src.cli.utils import console, error, info, parse_kv, success, warn
from src.iit.core.params import Params
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.runner import ejecutar, importar_estrategias
from src.infra.monitoring.resource_monitor import ResourceMonitor
from src.io.manager import cargar_mpt
from src.tui.run.csv_utils import CSV_HEADERS, cargar_indices_completados
from src.tui.run.helpers import (
    build_output_stem,
    etiqueta_estrategia,
    cargar_meta_programa,
    cargar_programa,
    guardar_meta_programa,
    guardar_programa,
    listar_estrategias,
)
from src.tui.run.platform import build_plataforma
from src.tui.test.generadores import etiquetas
from src.tui.test.helpers import cargar_patron, generar_combinaciones

OUTPUT_DIR = Path("data/output")


def handle(args) -> None:
    if args.resource != "execution":
        error("Solo se puede ejecutar 'execution'.")
        return

    name = args.name
    try:
        prog = cargar_programa(name)
    except FileNotFoundError:
        error(f"Execution '{name}' no encontrado.")
        return

    if not prog.dataset or not prog.patron or not prog.estrategia:
        error(
            f"Execution '{name}' incompleto: falta "
            f"{'dataset' if not prog.dataset else ''}"
            f"{'patrón' if not prog.patron else ''}"
            f"{'estrategia' if not prog.estrategia else ''}."
        )
        return

    if prog.estrategia not in listar_estrategias():
        error(f"Estrategia '{prog.estrategia}' no disponible.")
        return

    tpm = cargar_mpt(prog.dataset)
    patron = cargar_patron(prog.patron)
    n_dims = tpm.shape[1]
    combis = generar_combinaciones(patron, n_dims)
    total = len(combis)

    if total == 0:
        error("El patrón no genera combinaciones.")
        return

    opciones = dict(prog.opciones or {})
    if getattr(args, "opcion", None):
        try:
            opciones.update(parse_kv(args.opcion))
        except ValueError as e:
            error(str(e))
            return

    # Validar las opciones ANTES de arrancar: mejor fallar acá que en cada fila.
    if opciones:
        importar_estrategias()
        try:
            SIA.registry[prog.estrategia].validar_opciones(opciones)
        except ValueError as e:
            error(str(e))
            return
        info(f"Opciones: {', '.join(f'{k}={v}' for k, v in sorted(opciones.items()))}")

    # La etiqueta mete las opciones no-default en el nombre del CSV, para que
    # `qsw` y `qsw+modo=estatico` sean series distintas al comparar.
    output_stem = build_output_stem(
        name, prog.dataset, prog.patron, etiqueta_estrategia(prog.estrategia, opciones)
    )
    output_path = OUTPUT_DIR / f"{output_stem}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plataforma = build_plataforma()

    completados = cargar_indices_completados(output_path)
    meta = cargar_meta_programa(output_stem)
    params_actuales = {
        "dataset": prog.dataset,
        "patron": prog.patron,
        "estrategia": prog.estrategia,
        "opciones": opciones,
    }
    checkpoint_valido = bool(completados) and meta == params_actuales
    es_reanudacion = checkpoint_valido and not args.no_resume
    modo_apertura = "a" if es_reanudacion else "w"
    if not es_reanudacion:
        completados = set()
        guardar_meta_programa(output_stem, prog)

    prog.estado = "ejecutando"
    guardar_programa(prog)

    if es_reanudacion:
        info(f"Reanudando — {len(completados)}/{total} ya completados")
    else:
        info(f"Ejecutando {total} combinaciones...")

    # Barra de progreso Rich
    from rich.progress import Progress

    progress = Progress()
    task = progress.add_task(
        "[cyan]Ejecutando...[/cyan]", total=total, completed=len(completados)
    )

    cancelado = False
    filas = 0

    with progress:
        with output_path.open(modo_apertura, encoding="utf-8", newline="") as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            if not es_reanudacion:
                writer.writerow(CSV_HEADERS)

            for i, (estado, condicion, alcance, mecanismo) in enumerate(combis):
                if i in completados:
                    filas += 1
                    progress.update(task, advance=1)
                    continue

                if args.verbose:
                    combo_text = (
                        f"Test {i + 1}/{total}: "
                        f"({estado} | {condicion} | {alcance} | {mecanismo})"
                        f"  →  ({etiquetas(alcance)} | {etiquetas(mecanismo)})"
                    )
                    console.print(f"[dim]{combo_text}[/dim]")

                try:
                    params = Params(estado, condicion, alcance, mecanismo)
                    monitor = ResourceMonitor()
                    monitor.start()
                    sol = ejecutar(tpm, params, prog.estrategia, opciones)
                    stats = monitor.stop()
                    writer.writerow(
                        [
                            i,
                            estado,
                            condicion,
                            alcance,
                            mecanismo,
                            round(float(sol.perdida), 6),
                            stats.tiempo_wall_s,
                            stats.tiempo_cpu_s,
                            stats.cpu_user_s,
                            stats.cpu_sys_s,
                            stats.mem_rss_mb,
                            stats.gpu_mem_mb,
                            sol.particion.replace("\n", "\\n"),
                            plataforma,
                        ]
                    )
                    f.flush()
                except Exception as e:
                    warn(f"Fila {i} falló: {type(e).__name__}: {e}")

                filas += 1
                progress.update(task, advance=1)

    if not cancelado:
        prog.estado = "completado"
        prog.progreso = 100.0
        guardar_programa(prog)
        success(f"Completado — {filas}/{total} tests escritos en {output_path}")
    else:
        warn(f"Cancelado — {filas}/{total} completados")
