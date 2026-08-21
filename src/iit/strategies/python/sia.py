"""Estrategia base SIA: recibe el subsistema ya preparado y resuelve una solución óptima."""

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from src.iit.core.solution import Solution
from src.iit.core.system import System


class SIA(ABC):
    """Base para estrategias: operan sobre un subsistema y devuelven una Solution.

    El subsistema se obtiene fuera (reducir_a_subsistema(tpm, params)); la
    estrategia solo recibe ese System y en resolver() bipartir / buscar MIP
    y retornar la solución óptima.

    Registro automático
    -------------------
    Toda subclase que declare `nombre=` se registra en `SIA.registry` al
    importar su módulo. El runner llama a `importar_estrategias()` (que
    importa todos los code.py) antes de buscar en el registry, por lo que
    añadir una estrategia nueva solo requiere crear el directorio + code.py.

    Ejemplo de declaración:
        class MiEstrategia(SIA, nombre="mi_estrategia"):
        ...
        class Phi(SIA, nombre="phi", necesita_mpt=True):
        ...
    """

    registry: ClassVar[dict[str, type["SIA"]]] = {}
    necesita_mpt: ClassVar[dict[str, bool]] = {}

    #: Opciones configurables de la estrategia: ``{atributo: (valor, ...)}``.
    #: El primer valor de cada tupla es el default. La TUI renderiza un selector
    #: por opción y el CLI las acepta con ``--opcion attr=valor``; ``aplicar_opciones``
    #: las valida antes de asignarlas. Vacío = la estrategia no es configurable.
    opciones: ClassVar[dict[str, tuple[str, ...]]] = {}

    def __init_subclass__(
        cls, nombre: str = "", necesita_mpt: bool = False, **kw
    ) -> None:
        super().__init_subclass__(**kw)
        if nombre:
            cls.nombre = nombre
            SIA.registry[nombre] = cls
            SIA.necesita_mpt[nombre] = necesita_mpt

    @classmethod
    def defaults(cls) -> dict[str, str]:
        """Valor por defecto de cada opción (el primero de cada tupla)."""
        return {attr: valores[0] for attr, valores in cls.opciones.items()}

    @classmethod
    def validar_opciones(cls, opciones: dict[str, str] | None) -> dict[str, str]:
        """Valida opciones **sin instanciar** — sirve para fallar antes de arrancar.

        Lanza si el atributo no está declarado o el valor no es admisible: una
        opción mal escrita debe fallar, no ejecutarse en silencio con el default
        (el resultado quedaría etiquetado como algo que no corrió).
        """
        validadas = dict(opciones or {})
        for attr, valor in validadas.items():
            admisibles = cls.opciones.get(attr)
            if admisibles is None:
                raise ValueError(
                    f"'{cls.nombre}' no admite la opción {attr!r}. "
                    f"Disponibles: {sorted(cls.opciones) or 'ninguna'}"
                )
            if valor not in admisibles:
                raise ValueError(
                    f"{attr}={valor!r} inválido para '{cls.nombre}'. "
                    f"Valores: {', '.join(admisibles)}"
                )
        return validadas

    def aplicar_opciones(self, opciones: dict[str, str] | None) -> None:
        """Valida y asigna las opciones sobre esta instancia."""
        for attr, valor in type(self).validar_opciones(opciones).items():
            setattr(self, attr, valor)

    def __init__(self, subsistema: System) -> None:
        self.sistema: System = subsistema
        self.distribucion: NDArray[np.float32] = subsistema.distribucion_marginal()

    @abstractmethod
    def resolver(self) -> Solution:
        """Calcula la bipartición de mínima pérdida y retorna una Solution."""
