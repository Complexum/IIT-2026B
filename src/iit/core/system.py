from dataclasses import dataclass

import numpy as np

from src.iit.core.ncube import NCube


@dataclass(slots=True)
class System:
    estado_inicial: tuple[int, ...]
    ncubos: tuple[NCube, ...]

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(c.indice for c in self.ncubos)

    @property
    def dims(self) -> tuple[int, ...]:
        return self.ncubos[0].dims if self.ncubos else ()

    def condicionar(self, indices: tuple[int, ...]) -> "System":
        nuevos = tuple(
            c.condicionar(indices, self.estado_inicial)
            for c in self.ncubos
            if c.indice not in indices
        )
        return System(self.estado_inicial, nuevos)

    def substraer(
        self, alcance: tuple[int, ...], mecanismo: tuple[int, ...]
    ) -> "System":
        nuevos = tuple(
            c.marginalizar(mecanismo) for c in self.ncubos if c.indice not in alcance
        )
        return System(self.estado_inicial, nuevos)

    def bipartir(
        self, alcance: tuple[int, ...], mecanismo: tuple[int, ...]
    ) -> "System":
        nuevos = tuple(
            c.marginalizar(tuple(d for d in c.dims if d not in mecanismo))
            if c.indice in alcance
            else c.marginalizar(mecanismo)
            for c in self.ncubos
        )
        return System(self.estado_inicial, nuevos)

    def distribucion_marginal(self) -> np.ndarray:
        return np.array(
            [c[self.estado_inicial] for c in self.ncubos],
            dtype=np.float32,
        )

    def __str__(self) -> str:
        sub_dims = self.dims
        cubos_info = [f"{c}" for c in self.ncubos]
        return (
            f"\nSystem(indices={self.indices}, dims={sub_dims})"
            f"\nInitial state: {self.estado_inicial}"
            f"\nNCubes:\n" + "\n".join(cubos_info)
        )
