"""Parámetros de ejecución: estado inicial, condicionamiento, alcance y mecanismo.

Encapsulan lo necesario para construir candidato y subsistema a partir de un
sistema completo (TPM + estado). Se usan en Patrones y en Ejecución.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Parámetros para reducir sistema completo → candidato → subsistema.

    Todas las cadenas son de bits ('0' | '1') y deben tener la misma
    longitud (número de nodos/dimensiones).

    - estado: estado del sistema (ej. "1000000").
    - condicion: bits en 0 → dimensiones a condicionar (sacar del candidato).
    - alcance: bits en 0 → nodos a quitar del alcance (substraer).
    - mecanismo: bits en 0 → dimensiones a marginalizar en el mecanismo.
    """

    estado: str
    condicion: str
    alcance: str
    mecanismo: str

    def __post_init__(self) -> None:
        n = len(self.estado)
        valid = set("01")
        if not all(b in valid for b in self.estado):
            raise ValueError("estado inicial debe contener solo '0'/'1'")
        if len(self.condicion) != n or not all(b in valid for b in self.condicion):
            raise ValueError(
                "condicion debe tener misma longitud al estado inicial y solo '0'/'1'"
            )
        if len(self.alcance) != n or not all(b in valid for b in self.alcance):
            raise ValueError(
                "alcance debe tener misma longitud al estado inicial y solo '0'/'1'"
            )
        if len(self.mecanismo) != n or not all(b in valid for b in self.mecanismo):
            raise ValueError(
                "mecanismo debe tener misma longitud al estado inicial y solo '0'/'1'"
            )

    def estado_tuple(self) -> tuple[int, ...]:
        """Estado inicial como tupla de enteros para indexar TPM/NCubes."""
        return tuple(int(b) for b in self.estado)

    def dims_condicionar(self) -> tuple[int, ...]:
        """Índices de dimensiones a condicionar (bit en 0)."""
        return tuple(i for i, b in enumerate(self.condicion) if b == "0")

    def dims_alcance(self) -> tuple[int, ...]:
        """Índices a marginalziar del alcance (bit en 0) para substraer."""
        return tuple(i for i, b in enumerate(self.alcance) if b == "0")

    def dims_mecanismo(self) -> tuple[int, ...]:
        """Dimensiones del mecanismo a marginalizar (bit en 0)."""
        return tuple(i for i, b in enumerate(self.mecanismo) if b == "0")
