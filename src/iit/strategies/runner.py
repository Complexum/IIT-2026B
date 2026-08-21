"""Orquestación: importa estrategias, reduce sistema, ejecuta → Solution."""

import importlib
from pathlib import Path

import numpy as np

from src.iit.core.params import Params
from src.iit.core.solution import Solution
from src.iit.strategies.python.sia import SIA
from src.io.manager import reducir_a_subsistema

__STRATEGIES_DIR = Path("src/iit/strategies/python")


def importar_estrategias() -> None:
    """Importa todos los code.py de estrategias para que disparen SIA.__init_subclass__."""
    if not __STRATEGIES_DIR.exists():
        return
    for d in sorted(__STRATEGIES_DIR.iterdir()):
        if d.is_dir() and not d.name.startswith("__") and (d / "code.py").exists():
            modulo = f"src.iit.strategies.python.{d.name}.code"
            try:
                importlib.import_module(modulo)
            except Exception:
                # imports rotos no rompen el sistema — simplemente que no se registren
                pass


def ejecutar(
    tpm: np.ndarray,
    params: Params,
    nombre_estrategia: str,
    opciones: dict[str, str] | None = None,
) -> Solution:
    """Construye el subsistema desde (tpm, params), aplica la estrategia y devuelve la Solution.

    Args:
        opciones: overrides de atributos declarados en ``cls.opciones`` (ej.
            ``{"backend": "c", "modo": "estatico"}``). Se validan en
            ``SIA.aplicar_opciones``; una opción desconocida o un valor inválido
            lanzan en vez de caer al default.
    """
    importar_estrategias()

    if nombre_estrategia not in SIA.registry:
        disponibles = list(SIA.registry)
        raise ValueError(
            f"Estrategia desconocida: {nombre_estrategia!r}. Disponibles: {disponibles}"
        )

    cls = SIA.registry[nombre_estrategia]
    needs_tpm = SIA.necesita_mpt.get(nombre_estrategia, False)
    subsistema = reducir_a_subsistema(tpm, params)
    kwargs = {"tpm": tpm, "params": params} if needs_tpm else {}
    instancia = cls(subsistema, **kwargs)
    instancia.aplicar_opciones(opciones)
    return instancia.resolver()
