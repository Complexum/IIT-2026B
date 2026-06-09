"""CLI debug: corre phi y force en combinaciones explícitas sin checkpoint."""

from src.iit.core.params import Params
from src.iit.strategies.runner import ejecutar
from src.io.manager import cargar_mpt

DATASET = "N5B"
ESTADO = "10000"
CONDICION = "11111"

COMBOS = [
    ("11111", "11111"),  # full scope / full mec  — deben coincidir
    ("11111", "01110"),  # full scope / mec BCD   — diferente por fix
    ("11111", "01101"),  # full scope / mec BCD
    ("01110", "11111"),  # scope BCD  / full mec
    ("01110", "01110"),  # scope BCD  / mec BCD
]

tpm = cargar_mpt(DATASET)

print(f"{'strat':<8} {'alcance':<8} {'mec':<8} {'perdida':>10}  particion")
print("-" * 80)

for alcance, mec in COMBOS:
    for strat in ("phi", "analytic"):
        try:
            params = Params(ESTADO, CONDICION, alcance, mec)
            sol = ejecutar(tpm, params, strat)
            particion = sol.particion.replace("\n", " | ")
            print(f"{strat:<8} {alcance:<8} {mec:<8} {sol.perdida:>10.6f}  {particion}")
        except Exception as exc:
            print(f"{strat:<8} {alcance:<8} {mec:<8} {'ERROR':>10}  {exc}")
    print()
