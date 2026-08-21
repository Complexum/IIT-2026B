"""Orquestación: importa estrategias, reduce sistema, ejecuta → Solution."""

import importlib
from pathlib import Path

import numpy as np

from src.iit.core.params import Params
from src.iit.core.system import System
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


def preparar_subsistema(
    tpm: np.ndarray,
    params: Params,
    ncubos=None,
) -> System:
    """Sistema completo → subsistema. **No es parte del algoritmo.**

    Está separada de `resolver_estrategia` para que quien ejecuta pueda
    cronometrarlas aparte: la reducción cuesta lo mismo para todas las
    estrategias, así que meterla en la medición del algoritmo actúa como una
    constante compartida que comprime todos los speedups hacia 1. Medido sobre
    N20A + `patron-2`, escondía más de la mitad de la ventaja de `qsw+backend=c`
    (mostraba 1.55× donde el algoritmo va 3.43×).

    Args:
        ncubos: columnas ya preparadas (`io.manager.preparar_ncubos`). En un
            barrido, pasarlas evita rehacerlas por fila.
    """
    return reducir_a_subsistema(tpm, params, ncubos)


def resolver_estrategia(
    subsistema: System,
    nombre_estrategia: str,
    opciones: dict[str, str] | None = None,
    tpm: np.ndarray | None = None,
    params: Params | None = None,
) -> Solution:
    """Aplica la estrategia al subsistema ya preparado y devuelve la Solution.

    Esto —y sólo esto— es lo que hay que cronometrar para comparar estrategias.

    Args:
        opciones: overrides de atributos declarados en ``cls.opciones`` (ej.
            ``{"backend": "c", "modo": "estatico"}``). Se validan en
            ``SIA.aplicar_opciones``; una opción desconocida o un valor inválido
            lanzan en vez de caer al default.
        tpm, params: sólo para estrategias con ``necesita_mpt=True`` (``phi``).
    """
    importar_estrategias()

    if nombre_estrategia not in SIA.registry:
        disponibles = list(SIA.registry)
        raise ValueError(
            f"Estrategia desconocida: {nombre_estrategia!r}. Disponibles: {disponibles}"
        )

    cls = SIA.registry[nombre_estrategia]
    needs_tpm = SIA.necesita_mpt.get(nombre_estrategia, False)
    kwargs = {"tpm": tpm, "params": params} if needs_tpm else {}
    instancia = cls(subsistema, **kwargs)
    instancia.aplicar_opciones(opciones)
    return instancia.resolver()


def ejecutar(
    tpm: np.ndarray,
    params: Params,
    nombre_estrategia: str,
    opciones: dict[str, str] | None = None,
    ncubos=None,
) -> Solution:
    """Compone `preparar_subsistema` + `resolver_estrategia`.

    Conveniencia para quien no necesita cronometrar las dos fases por separado
    (``main.py``, tests). Los barridos de `cli run` y del tab Execution llaman a
    las dos funciones por separado para poder medir sólo el algoritmo.
    """
    # Validar el nombre antes de preparar: un typo no debería pagar la reducción
    # del subsistema para recién después fallar.
    importar_estrategias()
    if nombre_estrategia not in SIA.registry:
        raise ValueError(
            f"Estrategia desconocida: {nombre_estrategia!r}. "
            f"Disponibles: {list(SIA.registry)}"
        )
    subsistema = preparar_subsistema(tpm, params, ncubos)
    return resolver_estrategia(subsistema, nombre_estrategia, opciones, tpm, params)
