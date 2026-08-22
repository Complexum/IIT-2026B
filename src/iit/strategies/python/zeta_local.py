"""Oráculo por Zeta **local**: una transformada por nodo, sobre su propia tabla.

El precómputo actual (`zeta.py`) corre la Zeta sobre la TPM materializada: `D`
pasadas sobre un arreglo de `N × 2^D`, o sea `Θ(D·N·2^D)`. Ese `2^D` no viene del
problema sino de la **materialización**: un nodo con `k` padres tiene una tabla de
`2^k`, y expandirla a una columna de `2^D` la replica `2^(D-k)` veces.

Acá la Zeta corre sobre la tabla de cada nodo, de tamaño `2^kᵢ`::

    Θ(Σᵢ kᵢ · 2^kᵢ)     en vez de     Θ(D · N · 2^D)

Es **exacto**, no muestreado. Con N=D=20 y k=2 son 160 operaciones contra 4.2·10⁸.

Por qué funciona
----------------
`f(S) = Σᵢ |mean_{Fᵢ}(δᵢ)|`, y la media sobre una cara es un promedio sobre el
subcubo donde las dims libres varían y el resto queda fijo en el pivote. Como
`Hᵢ` **sólo depende de sus padres**, las dims libres que no son padres de `i` no
cambian su valor: promediar sobre ellas es promediar una constante. Entonces

    mean_A(δᵢ) = mean_{A ∩ padres(i)}(δᵢ)

y el promedio se toma sobre `2^|A ∩ padres(i)|` celdas, no sobre `2^|A|`. Toda la
cara global colapsa a una cara de la tabla local del nodo.
"""

from dataclasses import dataclass

import numpy as np

from src.io.fst import EspecFST
from src.iit.strategies.python.zeta import zeta_inplace


@dataclass(frozen=True)
class OraculoLocal:
    """Sumas de cara por nodo, cada una en coordenadas delta de su propia tabla."""

    #: `sumas[i]` tiene `2^kᵢ` entradas: la cara local `m` vive en `pivotes[i] ^ m`.
    sumas: tuple[np.ndarray, ...]
    #: Patrón del estado inicial restringido a los padres de cada nodo.
    pivotes: tuple[int, ...]
    #: `padres[i][j]` = posición de bit **global** (sobre `dims`) del padre local `j`.
    padres: tuple[tuple[int, ...], ...]
    D: int

    @property
    def coste_precomputo(self) -> int:
        """`Σ kᵢ·2^kᵢ` — con qué comparar el `D·N·2^D` de la ruta materializada."""
        return sum(max(len(p), 1) * (1 << len(p)) for p in self.padres)


def preparar_oraculo_local(espec: EspecFST, estado_inicial: tuple[int, ...]) -> OraculoLocal:
    """Zeta sobre la tabla de cada nodo. Nunca materializa la TPM.

    Args:
        espec: cualquier cosa con `.padres` y `.tablas` — un `EspecFST` (sistema
            completo) o un `reduccion.SubEspec` (subsistema, donde `D != N`).
        estado_inicial: valor de cada **dim** en `t`, indexado por posición de bit.
            Su longitud define `D`.
    """
    sumas, pivotes = [], []
    for i, padres in enumerate(espec.padres):
        k = len(padres)
        piv = sum((int(estado_inicial[p]) & 1) << j for j, p in enumerate(padres))

        # δ = H − p sobre la tabla local; el pivote queda en 0.
        # `np.array` COPIA: `ascontiguousarray` devolvería la tabla misma si ya es
        # contigua y float64, y el `-=` de abajo mutaría la spec en sitio.
        flat = np.array(espec.tablas[i], dtype=np.float64).reshape(1, 1 << k)
        flat -= flat[0, piv]
        if k:
            zeta_inplace(flat, 1, k, piv)

        sumas.append(flat[0])
        pivotes.append(piv)

    return OraculoLocal(
        sumas=tuple(sumas),
        pivotes=tuple(pivotes),
        padres=espec.padres,
        # D sale del estado, no del número de nodos: en un subsistema difieren.
        D=len(estado_inicial),
    )


def _mascara_local(masks: np.ndarray, padres: tuple[int, ...]) -> np.ndarray:
    """Máscara de mecanismo global → máscara sobre los padres del nodo. Little-endian."""
    local = np.zeros(masks.shape, dtype=np.int64)
    for j, p in enumerate(padres):
        local |= ((masks >> p) & 1) << j
    return local


def f_cara_batch_local(
    orc: OraculoLocal, masks_mec: np.ndarray, alc_bool: np.ndarray
) -> np.ndarray:
    """Misma interfaz que `qn.oracle.f_cara_batch`, sin tabla de `2^D`.

    Args:
        masks_mec: `(K,)` máscara de mecanismo por corte, sobre las dims globales.
        alc_bool: `(K, N)` bool, si el nodo `i` está en el alcance del corte `k`.
    """
    m = np.asarray(masks_mec, dtype=np.int64)
    en_alcance = np.asarray(alc_bool, dtype=bool)
    total = np.zeros(m.size, dtype=np.float64)

    for i, padres in enumerate(orc.padres):
        k = len(padres)
        if k == 0:
            continue  # nodo constante: δ ≡ 0, no aporta

        local = _mascara_local(m, padres)
        completo = (1 << k) - 1
        compl = completo ^ local
        piv = orc.pivotes[i]
        sumas = orc.sumas[i]

        # popcount de la máscara LOCAL: el promedio es sobre 2^|A ∩ padres|.
        pc = np.zeros(local.size, dtype=np.int64)
        for j in range(k):
            pc += (local >> j) & 1

        val_a = np.abs(sumas[piv ^ local]) / np.exp2(pc)
        val_b = np.abs(sumas[piv ^ compl]) / np.exp2(k - pc)
        total += np.where(en_alcance[:, i], val_b, val_a)

    return total
