"""Comando: run execution <nombre> [--no-resume]."""

import csv
import time
from pathlib import Path

from src.cli.utils import console, error, info, parse_kv, success, warn
from src.iit.core.params import Params
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.runner import (
    importar_estrategias,
    preparar_subsistema,
    resolver_estrategia,
)
from src.infra.monitoring.resource_monitor import ResourceMonitor
from src.io.manager import cargar_mpt, preparar_ncubos
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

# Fallos consecutivos del mismo tipo antes de abortar el barrido.
MAX_FALLOS_SEGUIDOS = 3


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
    # Columnas del TPM una sola vez: rehacerlas por fila era el 66 %
    # del tiempo del barrido (ver io.manager.preparar_ncubos).
    ncubos = preparar_ncubos(tpm)
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

    # Preflight ANTES de arrancar: valida opciones Y disponibilidad real (backend
    # sin compilar, dependencia ausente). Sin esto un fallo sistémico se descubre
    # fila por fila y el barrido termina "completado" con 0 resultados.
    importar_estrategias()
    try:
        SIA.registry[prog.estrategia].preflight(opciones)
    except Exception as e:
        error(str(e))
        return
    if opciones:
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
    fallos_seguidos = 0
    ultimo_error = ""

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

                    # La preparación NO es parte del algoritmo: cuesta lo mismo
                    # para todas las estrategias, así que se cronometra aparte y
                    # el monitor arranca recién después.
                    t_prep0 = time.perf_counter()
                    subsistema = preparar_subsistema(tpm, params, ncubos)
                    t_prep = time.perf_counter() - t_prep0

                    monitor = ResourceMonitor()
                    monitor.start()
                    sol = resolver_estrategia(
                        subsistema, prog.estrategia, opciones, tpm, params
                    )
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
                            round(t_prep, 6),
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
                    fallos_seguidos = 0
                except Exception as e:
                    warn(f"Fila {i} falló: {type(e).__name__}: {e}")
                    # Un fallo sistémico (no un dato raro) se repite idéntico. Cortar
                    # en vez de escupir un error por combinación y terminar con 0 filas.
                    if type(e).__name__ == ultimo_error:
                        fallos_seguidos += 1
                    else:
                        ultimo_error, fallos_seguidos = type(e).__name__, 1
                    if fallos_seguidos >= MAX_FALLOS_SEGUIDOS:
                        error(
                            f"{fallos_seguidos} fallos consecutivos de {ultimo_error} — "
                            f"abortando. Parece un problema de configuración, no de datos."
                        )
                        cancelado = True
                        break

                filas += 1
                progress.update(task, advance=1)

    if not cancelado:
        prog.estado = "completado"
        prog.progreso = 100.0
        guardar_programa(prog)
        success(f"Completado — {filas}/{total} tests escritos en {output_path}")
    else:
        warn(f"Cancelado — {filas}/{total} completados")
