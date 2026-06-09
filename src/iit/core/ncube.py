from dataclasses import dataclass

import numpy as np

# from iit.base.app import aplicacion
# aplicacion


@dataclass(frozen=True, slots=True)
class NCube:
    """
    N-cubo: estructura n-dimensional aplanada donde cada dimensión es binaria.

    - `indice`: nodo asociado (0→A, 1→B, 2→C, ...).
    - `dims`: dimensiones activas, ordenadas ascendente e inmutables.
    - `data`: arreglo plano de longitud 2^len(dims).
      El bit en posición j del índice plano corresponde a dims[j].
    """

    indice: int
    dims: tuple[int, ...]
    data: np.ndarray

    @property
    def ndata(self) -> np.ndarray:
        """Data como N-dimensional array, axis d = dims[d].
        ndata[j0, j1, ...] = P(node=1 | dims[0]=j0, dims[1]=j1, ...)
        """
        return self.data.reshape((2,) * len(self.dims), order="F")

    def idx(self, dim: int) -> int | None:
        try:
            return self.dims.index(dim)
        except ValueError:
            return None

    def condicionar(
        self,
        indices: tuple[int, ...],
        estado: tuple[int, ...],
    ) -> "NCube":
        """
        Fija dimensiones según estado inicial, descartando las caras no seleccionadas.

        Para cada dim en `indices` que exista en este cubo, se fija su bit
        al valor de estado[dim], filtrando los elementos del arreglo plano
        que no coincidan.

        Ejemplo:
            dims=(0,1,2), estado=(1,0,0), condicionar en (2,)
            → Fija dim2=0, queda dims=(0,1) con las caras donde dim2=0.
        """
        fijos: list[tuple[int, int]] = []
        for dim in indices:
            pos = self.idx(dim)
            if pos is not None:
                fijos.append((pos, estado[dim] & 1))

        if not fijos:
            return self

        nuevas_dims = tuple(d for d in self.dims if d not in indices)

        todos = np.arange(len(self.data), dtype=np.intp)
        mascara = np.ones(len(self.data), dtype=bool)
        for pos, bit in fijos:
            mascara &= ((todos >> pos) & 1) == bit

        return NCube(self.indice, nuevas_dims, self.data[mascara])

    def marginalizar(self, ejes: tuple[int, ...]) -> "NCube":
        """
        Colapsa dimensiones promediando emparejando de valores (mantiene probabilidad).

        Para cada eje a marginalizar, se promedian todos los estados posibles
        de esas dimensiones, reduciendo la dimensionalidad del cubo.

        Ejemplo:
            dims=(0,1,2), marginalizar (1,2)
            - Promedia sobre combinaciones de dim1 y dim2, queda dims=(0,).
        """
        mpos = sorted({p for e in ejes if (p := self.idx(e)) is not None})

        if not mpos:
            return self

        nuevas_dims = tuple(d for d in self.dims if d not in ejes)

        k = len(self.dims)
        m = len(mpos)
        nuevo_k = k - m
        rpos = [p for p in range(k) if p not in mpos]

        denom = 1 << m
        nuevo_len = 1 << nuevo_k

        # Vectorizado: construir indices base con bits que se mantienen
        idx_salida = np.arange(nuevo_len, dtype=np.intp)
        idx_base = np.zeros(nuevo_len, dtype=np.intp)
        for j, pos in enumerate(rpos):
            idx_base |= ((idx_salida >> j) & 1) << pos

        combs = np.arange(denom, dtype=np.intp)
        bits_marg_all = np.zeros(denom, dtype=np.intp)
        for j, pos in enumerate(mpos):
            bits_marg_all |= ((combs >> j) & 1) << pos

        idx_all = idx_base[:, None] | bits_marg_all[None, :]  # (nuevo_len, 2^m)
        suma = self.data[idx_all].sum(axis=1)

        return NCube(self.indice, nuevas_dims, suma / denom)

    def __getitem__(self, estado: tuple[int, ...]):
        """Obtiene el valor puntual indexando por estado de cada dimensión."""
        if not self.dims:
            return float(self.data[0])
        idx = sum((estado[dim] & 1) << pos for pos, dim in enumerate(self.dims))
        return float(self.data[idx])

    def __str__(self) -> str:
        forma = (2,) * len(self.dims)
        if not self.dims:
            datos_str = str(self.data[0])
        else:
            datos_str = str(self.data.reshape(forma))
            datos_str = datos_str.replace("\n", "\n" + " " * 8)
        return (
            f"NCube(indice={self.indice}):\n"
            f"    dims={self.dims}\n"
            f"    shape={forma}\n"
            f"    data=\n        {datos_str}"
        )
