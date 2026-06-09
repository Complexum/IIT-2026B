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

    def __init_subclass__(
        cls, nombre: str = "", necesita_mpt: bool = False, **kw
    ) -> None:
        super().__init_subclass__(**kw)
        if nombre:
            cls.nombre = nombre
            SIA.registry[nombre] = cls
            SIA.necesita_mpt[nombre] = necesita_mpt

    def __init__(self, subsistema: System) -> None:
        self.sistema: System = subsistema
        self.distribucion: NDArray[np.float32] = subsistema.distribucion_marginal()

    @abstractmethod
    def resolver(self) -> Solution:
        """Calcula la bipartición de mínima pérdida y retorna una Solution."""
