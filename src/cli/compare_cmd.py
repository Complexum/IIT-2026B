"""Comando: compare <resultado> [<resultado> ...] [--plot].

Compara resultados de distintas estrategias sobre el mismo (dataset, patrón).
Espeja el flujo del tab Analysis de la TUI (`src/tui/analysis/screen.py`), pero
sin abrir la TUI: reutiliza `compare_group_n`, `build_rich_table_n`,
`generate_analysis_page` y `open_plot` sin modificarlos.

Por defecto imprime sólo los números. Con `--plot` genera además la página HTML
interactiva (plotly) y la abre; con `--paper`, las figuras matplotlib del paper.
"""

from pathlib import Path

from src.cli.utils import console, error, info, success, warn
from src.io.manager import listar_resultados
from src.tui.analysis.compare import (
    build_rich_table_n,
    compare_group_n,
    formatos_mezclados,
    group_selected,
    parse_result_key,
)

OUTPUT_DIR = Path("data/output")
PICS_DIR = Path("src/pics")


def _resolver(nombre: str, disponibles: list[str]) -> list[str]:
    """Un nombre de execution ('program-04') o una ruta completa de resultado."""
    if nombre in disponibles:
        return [nombre]
    coincidencias = [r for r in disponibles if r.startswith(f"{nombre}/")]
    if not coincidencias:
        error(f"No se encontró resultado para '{nombre}'.")
    return coincidencias


def _seleccionar(args, disponibles: list[str]) -> list[str]:
    if args.all:
        nombres = list(disponibles)
        if args.dataset:
            nombres = [n for n in nombres if parse_result_key(n)[0] == args.dataset]
        if args.patron:
            nombres = [n for n in nombres if parse_result_key(n)[2] == args.patron]
        return nombres
    nombres: list[str] = []
    for n in args.names:
        nombres.extend(_resolver(n, disponibles))
    return nombres


def handle(args) -> None:
    disponibles = listar_resultados()
    if not disponibles:
        error(f"No hay resultados en {OUTPUT_DIR}/.")
        return

    nombres = _seleccionar(args, disponibles)
    if len(nombres) < 2:
        error(f"Se necesitan 2+ resultados para comparar ({len(nombres)} encontrado(s)).")
        if not args.all:
            info("Disponibles:")
            for r in disponibles:
                console.print(f"  - {r}")
        return

    grupos = group_selected(nombres)

    # Los CSV anteriores al cambio de medición incluyen la preparación del
    # subsistema en `tiempo_wall_s`; los nuevos no. Los `perdida` son comparables
    # igual, pero los tiempos no — avisar en vez de dejar sacar conclusiones malas.
    viejos = formatos_mezclados(nombres)
    if viejos:
        warn(
            "Formatos de medición mezclados: en "
            + ", ".join(viejos)
            + " el tiempo incluye la preparación del subsistema; en el resto no. "
            "Los φ son comparables; los tiempos no. Re-ejecutá con --no-resume "
            "para homogeneizar."
        )

    tol = args.tol
    info(
        f"{len(nombres)} resultado(s) · {len(grupos)} grupo(s) · tol={tol:.0e}\n"
    )

    hubo_comparacion = False
    for (dataset, patron), names in grupos.items():
        console.print(f"[bold cyan]{dataset} | {patron}[/bold cyan]")
        if len(names) < 2:
            warn(f"  Sólo 1 estrategia ({parse_result_key(names[0])[1]}) — nada que comparar.")
            continue
        try:
            res = compare_group_n(names, tol)
        except Exception as exc:
            error(f"  {type(exc).__name__}: {exc}")
            continue
        console.print(f"  Estrategias: {', '.join(res['strategies'])}")
        console.print(build_rich_table_n(res["pairs"], res["strategies"], tol))
        hubo_comparacion = True

    if not hubo_comparacion:
        warn("Ningún grupo tenía 2+ estrategias. ¿Mismo dataset y patrón?")
        return

    if not (args.plot or args.paper):
        return

    if args.plot:
        try:
            from src.tui.analysis.plots import generate_analysis_page, open_plot

            ruta = generate_analysis_page(grupos, args.ref, tol)
        except ImportError:
            error("plotly no disponible — instalar con: uv add plotly")
            return
        except Exception as exc:
            error(f"Error generando la página: {type(exc).__name__}: {exc}")
            return
        success(f"Página de análisis → {ruta}")
        if not args.no_open:
            open_plot(ruta)

    if args.paper:
        try:
            from src.io.plots import generate_paper_figures

            rutas = generate_paper_figures(nombres, OUTPUT_DIR, PICS_DIR, args.ref)
        except Exception as exc:
            error(f"Error generando figuras del paper: {type(exc).__name__}: {exc}")
            return
        if not rutas:
            warn("Sin datos para las figuras del paper.")
            return
        success(f"Figuras del paper → {', '.join(str(p) for p in rutas)}")
