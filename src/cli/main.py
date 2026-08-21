"""Punto de entrada del CLI."""

import argparse
import sys

from src.cli import (
    compare_cmd,
    delete_cmd,
    edit_cmd,
    list_cmd,
    new_cmd,
    results_cmd,
    run_cmd,
    show_cmd,
)
from src.cli.utils import console, error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iit",
        description="IIT-2026A CLI — ejecuta sin TUI",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Salida detallada (verbose)"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ── list ────────────────────────────────────────────────
    list_parser = sub.add_parser("list", help="Listar recursos")
    list_parser.add_argument(
        "resource",
        choices=["datasets", "patterns", "executions", "strategies"],
        help="Tipo de recurso a listar",
    )
    list_parser.set_defaults(func=list_cmd.handle)

    # ── show ────────────────────────────────────────────────
    show_parser = sub.add_parser("show", help="Mostrar detalles de un recurso")
    show_parser.add_argument(
        "resource",
        choices=["dataset", "execution", "patron", "strategy"],
        help="Tipo de recurso",
    )
    show_parser.add_argument("name", help="Nombre del recurso")
    show_parser.set_defaults(func=show_cmd.handle)

    # ── new ─────────────────────────────────────────────────
    new_parser = sub.add_parser("new", help="Crear un nuevo recurso")
    new_parser.add_argument(
        "resource",
        choices=["dataset", "execution", "patron"],
        help="Tipo de recurso a crear",
    )
    new_parser.add_argument(
        "name_or_dims", help="Nombre del recurso o dimensiones (dataset)"
    )
    # Campos opcionales compartidos para execution / patron
    new_parser.add_argument(
        "--dataset", dest="dataset", default="", help="Dataset (para execution)"
    )
    new_parser.add_argument(
        "--patron", dest="patron", default="", help="Patrón (para execution)"
    )
    new_parser.add_argument(
        "--estrategia",
        dest="estrategia",
        default="",
        help="Estrategia (para execution)",
    )
    new_parser.add_argument(
        "--discretos", action="store_true", help="Datos deterministas (dataset)"
    )
    new_parser.add_argument(
        "--opcion", action="append", default=[], metavar="ATTR=VALOR", help="Opción de la estrategia, repetible (ej: --opcion modo=estatico --opcion backend=c). Ver `cli show strategy <nombre>`."
    )
    new_parser.set_defaults(func=new_cmd.handle)

    # ── edit ────────────────────────────────────────────────
    edit_parser = sub.add_parser("edit", help="Editar un recurso existente")
    edit_parser.add_argument(
        "resource",
        choices=["execution"],
        help="Tipo de recurso a editar",
    )
    edit_parser.add_argument("name", help="Nombre del recurso")
    edit_parser.add_argument("--dataset", dest="dataset", default=None, help="Dataset")
    edit_parser.add_argument("--patron", dest="patron", default=None, help="Patrón")
    edit_parser.add_argument(
        "--estrategia", dest="estrategia", default=None, help="Estrategia"
    )
    edit_parser.add_argument(
        "--opcion", action="append", default=[], metavar="ATTR=VALOR", help="Opción de la estrategia, repetible (ej: --opcion modo=estatico --opcion backend=c). Ver `cli show strategy <nombre>`."
    )
    edit_parser.set_defaults(func=edit_cmd.handle)

    # ── run ─────────────────────────────────────────────────
    run_parser = sub.add_parser("run", help="Ejecutar un execution")
    run_parser.add_argument(
        "resource",
        choices=["execution"],
        help="Tipo de recurso a ejecutar",
    )
    run_parser.add_argument("name", help="Nombre del execution")
    run_parser.add_argument(
        "--no-resume", action="store_true", help="No reanudar; empezar de cero"
    )
    run_parser.add_argument(
        "--opcion", action="append", default=[], metavar="ATTR=VALOR", help="Opción de la estrategia, repetible (ej: --opcion modo=estatico --opcion backend=c). Ver `cli show strategy <nombre>`."
    )
    run_parser.set_defaults(func=run_cmd.handle)

    # ── results ────────────────────────────────────────────
    results_parser = sub.add_parser("results", help="Consultar resultados con SQL")
    results_parser.add_argument(
        "name",
        help="Nombre del execution o ruta completa del resultado (ej: program-01 o program-01/N10B--phi--patron-2)",
    )
    results_parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Query SQL opcional (ej: 'SELECT estado, perdida WHERE perdida > 0.5 ORDER BY tiempo DESC LIMIT 10')",
    )
    results_parser.set_defaults(func=results_cmd.handle)

    # ── compare ────────────────────────────────────────────
    cmp_parser = sub.add_parser(
        "compare", help="Comparar resultados de varias estrategias"
    )
    cmp_parser.add_argument(
        "names",
        nargs="*",
        help="Executions o rutas de resultado (ej: program-04 program-10)",
    )
    cmp_parser.add_argument(
        "--all", action="store_true", help="Comparar todos los resultados disponibles"
    )
    cmp_parser.add_argument("--dataset", default=None, help="Filtrar por dataset (con --all)")
    cmp_parser.add_argument("--patron", default=None, help="Filtrar por patrón (con --all)")
    cmp_parser.add_argument(
        "--tol", type=float, default=1e-4, help="Tolerancia de igualdad (default 1e-4)"
    )
    cmp_parser.add_argument(
        "--ref", default=None, help="Estrategia de referencia para el error relativo"
    )
    cmp_parser.add_argument(
        "--plot", action="store_true", help="Generar la página HTML interactiva (plotly)"
    )
    cmp_parser.add_argument(
        "--paper", action="store_true", help="Generar las figuras del paper (matplotlib)"
    )
    cmp_parser.add_argument(
        "--no-open", action="store_true", help="No abrir el navegador"
    )
    cmp_parser.set_defaults(func=compare_cmd.handle)

    # ── delete ─────────────────────────────────────────────
    del_parser = sub.add_parser("delete", help="Eliminar un recurso")
    del_parser.add_argument(
        "resource",
        choices=["dataset", "execution", "patron"],
        help="Tipo de recurso a eliminar",
    )
    del_parser.add_argument("name", help="Nombre del recurso")
    del_parser.set_defaults(func=delete_cmd.handle)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Cancelado por usuario.[/bold yellow]")
        sys.exit(130)
    except Exception as e:
        error(f"{type(e).__name__}: {e}")
        if args.verbose:
            console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
