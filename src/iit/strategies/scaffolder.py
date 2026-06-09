"""Scaffolding de nueva estrategia IIT.

Uso:
    uv run estrategia mi_algoritmo
    uv run estrategia mi_fuerza_bruta --tpm
"""

import argparse
import sys
from pathlib import Path

STRATEGIES_DIR = Path("src/iit/strategies/python")

_TEMPLATE = '''\
"""Estrategia {nombre_cap}: <descripción breve>."""

import time

from src.iit.core.solution import Solution
from src.iit.strategies.python.func import Parte, fmt_particion
from src.iit.strategies.python.sia import SIA
from src.infra.middlewares.profile import perfilar


@perfilar
class {clase}(SIA, nombre="{nombre}"{necesita_mpt_arg}):
    """Descripción de la estrategia."""
{init_extra}
    def resolver(self) -> Solution:
        dm_original = self.distribucion
        sistema = self.sistema  # noqa: F841  — System listo para bipartir
        t0 = time.perf_counter()

        # TODO: implementar el algoritmo de partición mínima
        # Referencia: sistema.bipartir(alcance, mecanismo) → System
        #             sistema.distribucion_marginal() → NDArray
        #             fmt_particion(Parte(num, den), ...) → str
        mejor_perdida = 0.0
        mejor_dist = dm_original
        particion_str = fmt_particion(Parte("\\u2205", "\\u2205"), Parte("\\u2205", "\\u2205"))

        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=mejor_perdida,
            distribucion_subsistema=dm_original,
            distribucion_particion=mejor_dist,
            particion=particion_str.strip(),
            tiempo_total=time.perf_counter() - t0,
            quiere_hablar=False,
        )
'''

_INIT_TPM = """\

    def __init__(self, subsistema, tpm):
        super().__init__(subsistema)
        self.tpm = tpm  # TPM completa (2^n, n) — disponible en resolver()
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="estrategia",
        description="Crear scaffolding de nueva estrategia IIT",
    )
    parser.add_argument("nombre", help="Nombre de la estrategia (snake_case o kebab-case)")
    parser.add_argument(
        "--tpm",
        action="store_true",
        help="La estrategia requiere la TPM completa (necesita_mpt=True, como pyphi)",
    )
    args = parser.parse_args()

    nombre = args.nombre.lower().replace("-", "_")
    clase = "".join(p.capitalize() for p in nombre.split("_"))

    dest = STRATEGIES_DIR / nombre
    if dest.exists():
        print(f"✗  Ya existe: {dest}")
        sys.exit(1)

    dest.mkdir(parents=True)
    (dest / "__init__.py").write_text("", encoding="utf-8")
    (dest / "code.py").write_text(
        _TEMPLATE.format(
            nombre=nombre,
            nombre_cap=nombre.replace("_", " ").capitalize(),
            clase=clase,
            necesita_mpt_arg=", necesita_mpt=True" if args.tpm else "",
            init_extra=_INIT_TPM if args.tpm else "",
        ),
        encoding="utf-8",
    )

    print(f"✓  Estrategia '{nombre}' creada:")
    print(f"   {dest}/__init__.py")
    print(f"   {dest}/code.py")
    print()
    print("   Editar resolver() para implementar el algoritmo.")
    print("   La estrategia aparece automáticamente en la TUI y en ejecutar().")
    if args.tpm:
        print("   self.tpm disponible en resolver() (TPM completa).")


if __name__ == "__main__":
    main()
