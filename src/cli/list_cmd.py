"""Comando: list datasets | patterns | executions | strategies."""

from src.cli.utils import print_table, warn
from src.iit.strategies.runner import SIA
from src.io.manager import listar_redes
from src.tui.run.helpers import listar_programas
from src.tui.test.helpers import listar_patrones


def handle(args) -> None:
    resource = args.resource

    if resource == "datasets":
        items = listar_redes()
        if not items:
            warn("No hay datasets disponibles.")
            return
        print_table(["Dataset"], [[n] for n in items], title="Datasets")

    elif resource == "patterns":
        items = listar_patrones()
        if not items:
            warn("No hay patrones disponibles.")
            return
        print_table(["Patrón"], [[n] for n in items], title="Patrones")

    elif resource == "executions":
        items = listar_programas()
        if not items:
            warn("No hay executions disponibles.")
            return
        rows = []
        for n in items:
            prog = __import__(
                "src.tui.run.helpers", fromlist=["cargar_programa"]
            ).cargar_programa(n)
            rows.append(
                [
                    n,
                    prog.dataset or "—",
                    prog.patron or "—",
                    prog.estrategia or "—",
                    prog.estado,
                ]
            )
        print_table(
            ["Execution", "Dataset", "Patrón", "Estrategia", "Estado"],
            rows,
            title="Executions",
        )

    elif resource == "strategies":
        # Importar estrategias para poblar el registro
        from src.iit.strategies.runner import __importar_estrategias

        __importar_estrategias()
        items = sorted(SIA.registry.keys())
        if not items:
            warn("No hay estrategias disponibles.")
            return
        rows = [[n, "Sí" if SIA.necesita_mpt.get(n, False) else "No"] for n in items]
        print_table(["Estrategia", "Necesita MPT"], rows, title="Estrategias")

    else:
        warn(f"Recurso desconocido: {resource}")
