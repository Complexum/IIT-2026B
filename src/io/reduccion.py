"""Reducir una especificación a un subsistema, sin materializar la TPM.

Réplica de `manager.reducir_a_subsistema` (sistema completo → condicionar →
substraer) pero operando sobre las **tablas locales** de cada nodo, que tienen
`2^kᵢ` celdas en vez de `2^D`. Las tres operaciones son `O(2^kᵢ)`:

| operación | sobre la fórmula CNF | sobre la tabla |
|---|---|---|
| condicionar `d = v` | **propagación unitaria** (`cnf.condicionar`) — CNF → CNF | slice de la mitad |
| marginalizar `d` | `½·f\\|_{d=0} + ½·f\\|_{d=1}` — CNF → **PBN** | promedio de las dos mitades |
| substraer nodo | quitar su entrada | quitar la fila |

Nótese la segunda: marginalizar **no es cerrado sobre funciones booleanas**, porque
el promedio de dos booleanas no es booleana. Pero es exactamente una mezcla de dos
con peso ½, o sea un PBN. **El PBN es la clausura algebraica de CNF bajo la
marginalización**, que es justo la operación que la reducción de IIT necesita — una
razón para el modelo bastante mejor que "sirve para meter ruido".

Acá se opera sobre la tabla y no sobre la fórmula porque la tabla ya es el compilado
que el oráculo lee, y las dos rutas coinciden (verificado en `tests/dataset/`).
"""

from dataclasses import dataclass

import numpy as np

from src.iit.core.params import Params
from src.io.fst import EspecFST


@dataclass(frozen=True)
class SubEspec:
    """Subsistema: `D` dims y `N` nodos, que ya **no** tienen por qué coincidir.

    `padres[i]` son **posiciones dentro de `dims`**, no etiquetas originales: es lo
    que el oráculo necesita, porque los bits de la máscara de mecanismo indexan
    `dims`. `dims` e `indices` guardan las etiquetas originales para poder reportar
    la partición ganadora.
    """

    dims: tuple[int, ...]
    indices: tuple[int, ...]
    padres: tuple[tuple[int, ...], ...]
    tablas: tuple[np.ndarray, ...]
    estado: tuple[int, ...]

    def distribucion_marginal(self) -> np.ndarray:
        """Igual que `System.distribucion_marginal`.

        Existe para que `SIA.__init__` acepte un `SubEspec` sin cambios: lo único
        que la clase base pide de un subsistema es esto, más `dims` e `indices`.
        """
        return marginal(self)

    def bipartir_marginal(
        self, alcance: set[int] | frozenset[int], mecanismo: set[int] | frozenset[int]
    ) -> np.ndarray:
        """`System.bipartir(...).distribucion_marginal()` en un paso, por posiciones."""
        return marginal_biparticion(self, alcance, mecanismo)

    def __post_init__(self) -> None:
        if len(self.estado) != len(self.dims):
            raise ValueError("estado debe tener una entrada por dim")
        if len(self.padres) != len(self.indices) or len(self.tablas) != len(self.indices):
            raise ValueError("padres y tablas deben tener una entrada por nodo")
        for i, (ps, t) in enumerate(zip(self.padres, self.tablas)):
            if len(t) != (1 << len(ps)):
                raise ValueError(f"nodo {i}: tabla de {len(t)} para {len(ps)} padres")


def condicionar_tabla(
    padres: tuple[int, ...], tabla: np.ndarray, dim: int, valor: int
) -> tuple[tuple[int, ...], np.ndarray]:
    """Fija el padre `dim` en `valor`: se queda con la mitad de la tabla. `O(2^k)`."""
    if dim not in padres:
        return padres, tabla
    j = padres.index(dim)
    idx = np.arange(tabla.size, dtype=np.int64)
    return (
        tuple(p for p in padres if p != dim),
        tabla[((idx >> j) & 1) == valor],
    )


def marginalizar_tabla(
    padres: tuple[int, ...], tabla: np.ndarray, dim: int
) -> tuple[tuple[int, ...], np.ndarray]:
    """Promedia las dos mitades a lo largo del padre `dim`. `O(2^k)`.

    Es la operación que saca del mundo booleano: el resultado es una mezcla, o sea
    un nodo PBN aunque el original fuera determinista.
    """
    if dim not in padres:
        return padres, tabla
    j = padres.index(dim)
    idx = np.arange(tabla.size, dtype=np.int64)
    bajo = idx[((idx >> j) & 1) == 0]
    return (
        tuple(p for p in padres if p != dim),
        (tabla[bajo] + tabla[bajo | (1 << j)]) / 2.0,
    )


def reducir_espec(espec: EspecFST, params: Params) -> SubEspec:
    """Sistema completo → candidato (condicionar) → subsistema (substraer).

    Espeja `manager.reducir_a_subsistema` paso a paso, así que el subsistema
    resultante debe dar el mismo oráculo — pero sin construir nunca la TPM.
    """
    estado = params.estado_tuple()
    cond = set(params.dims_condicionar())
    quitar_nodos = set(params.dims_alcance())
    marginar = set(params.dims_mecanismo())

    indices = tuple(
        i for i in range(espec.n) if i not in cond and i not in quitar_nodos
    )
    # `System.dims` sale del primer cubo sobreviviente, así que sin nodos no hay dims.
    dims = (
        tuple(d for d in range(espec.n) if d not in cond and d not in marginar)
        if indices
        else ()
    )

    padres_out, tablas_out = [], []
    for i in indices:
        ps, tabla = espec.padres[i], espec.tablas[i]
        for d in sorted(cond):  # condicionar: fija la dim en su valor del estado
            ps, tabla = condicionar_tabla(ps, tabla, d, int(estado[d]) & 1)
        for d in sorted(marginar):  # substraer: promedia sobre la dim
            ps, tabla = marginalizar_tabla(ps, tabla, d)
        # Reetiquetar a posiciones dentro de `dims`, que es lo que indexa la máscara.
        pos = {d: j for j, d in enumerate(dims)}
        padres_out.append(tuple(pos[d] for d in ps))
        tablas_out.append(tabla)

    return SubEspec(
        dims=dims,
        indices=indices,
        padres=tuple(padres_out),
        tablas=tuple(tablas_out),
        estado=tuple(int(estado[d]) & 1 for d in dims),
    )


def _media_cara(
    padres: tuple[int, ...], tabla: np.ndarray, estado: tuple[int, ...], libres: set[int]
) -> float:
    """Media de la tabla con las dims de `libres` variando y el resto en `estado`.

    Es `NCube.marginalizar(libres)` seguido de evaluar en el estado, en un solo
    paso y en `O(2^k)`. Las dims de `libres` que no son padres del nodo no cambian
    su valor, así que promediar sobre ellas es promediar una constante y se ignoran.
    """
    libres_locales = [j for j, p in enumerate(padres) if p in libres]
    fijos = sum(
        (estado[p] & 1) << j for j, p in enumerate(padres) if p not in libres
    )
    if not libres_locales:
        return float(tabla[fijos])

    u = np.arange(1 << len(libres_locales), dtype=np.int64)
    despl = np.zeros(u.size, dtype=np.int64)
    for t, j in enumerate(libres_locales):
        despl |= ((u >> t) & 1) << j
    return float(tabla[fijos | despl].mean())


def marginal_biparticion(
    sub: SubEspec,
    alcance: set[int] | frozenset[int],
    mecanismo: set[int] | frozenset[int],
) -> np.ndarray:
    """Equivalente de `System.bipartir(...).distribucion_marginal()`, sin `System`.

    Args:
        alcance: **posiciones** de nodo (0..N−1) que quedan del lado del alcance.
        mecanismo: **posiciones** de dim (0..D−1) del mecanismo.

    Reproduce la regla de `System.bipartir`: un nodo del alcance marginaliza las
    dims que **no** están en el mecanismo; el resto marginaliza las que sí. Con
    ambos conjuntos vacíos devuelve la marginal sin partir, o sea el valor de cada
    nodo en el estado actual.
    """
    mec = set(mecanismo)
    complemento = set(range(len(sub.dims))) - mec
    valores = [
        _media_cara(ps, tabla, sub.estado, complemento if i in alcance else mec)
        for i, (ps, tabla) in enumerate(zip(sub.padres, sub.tablas))
    ]
    return np.array(valores, dtype=np.float32)


def marginal(sub: SubEspec) -> np.ndarray:
    """Valor de cada nodo en el estado actual — la marginal sin partir."""
    return marginal_biparticion(sub, frozenset(), frozenset())
