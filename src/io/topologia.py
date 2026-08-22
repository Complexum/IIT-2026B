"""Topología de la red: quién es padre de quién, y si eso la vuelve trivial.

Separado de `fst.py` porque es una responsabilidad propia y porque el chequeo de
separabilidad es lo que decide si una red sirve como caso de prueba.

Por qué importa la separabilidad
---------------------------------
El costo de un corte es `f(S) = Σᵢ |mean_{Fᵢ}(δᵢ)|`. Si `A ∩ padres(i) = ∅`, la cara
del nodo `i` colapsa a la celda del pivote, donde `δᵢ = 0` **por construcción**: el
término se anula sin importar qué tenga la tabla.

Entonces `f(S) = 0` en cuanto exista un **corte cerrado** — un conjunto de dims `A`
tal que cada nodo tenga *todos* o *ninguno* de sus padres en `A`. Se pone en el
alcance a los nodos con todos sus padres en `A` y afuera al resto.

Eso es **topológico**: no depende de los valores de la tabla, así que ni las
probabilidades ni el ruido lo arreglan (verificado: binario, PBN y ruido dan
`f = 0.0000000000` sobre el mismo corte). Y no es un error de cálculo — una red que
se separa limpio **no está integrada** y `φ = 0` es la respuesta correcta de IIT.

Sorteando padres uniformemente con `k` chico casi siempre aparece un corte cerrado,
que es justo el régimen donde el oráculo local gana. De ahí `modo="anillo"`.
"""

import numpy as np

#: Modos de topología disponibles.
MODOS = ("aleatoria", "anillo")


def elegir_padres(
    n: int,
    k: int,
    rng: np.random.Generator,
    *,
    auto_lazo: bool = True,
    modo: str = "aleatoria",
) -> tuple[tuple[int, ...], ...]:
    """`k` padres por nodo. Compartida por el generador binario y el PBN.

    Args:
        auto_lazo: si el nodo puede ser padre de sí mismo (el estado de un nodo en
            t+1 depende del suyo en t — semántica normal de un FST). Ignorado en
            `modo="anillo"`, donde el auto-lazo es parte de la construcción.
        modo: `"aleatoria"` sortea los `k` padres uniformemente — dispersa pero casi
            siempre **separable**, o sea `φ = 0`. `"anillo"` fija `{i−1, i}` como
            columna vertebral y sortea los `k−2` restantes: la clausura de cualquier
            dim recorre el ciclo entero, así que **no existe corte cerrado propio**.
    """
    if n < 1:
        raise ValueError("n debe ser positivo")
    if modo not in MODOS:
        raise ValueError(f"modo desconocido: {modo!r} (usar {list(MODOS)})")
    k = max(1, min(k, n))

    if modo == "anillo":
        if k < 2:
            raise ValueError("el anillo necesita k >= 2 para tener columna vertebral")
        salida = []
        for i in range(n):
            fijos = {(i - 1) % n, i}
            resto = [d for d in range(n) if d not in fijos]
            extra = rng.choice(resto, size=min(k - 2, len(resto)), replace=False)
            salida.append(tuple(sorted(fijos | set(int(x) for x in extra))))
        return tuple(salida)

    salida = []
    for i in range(n):
        elegibles = np.arange(n) if auto_lazo else np.delete(np.arange(n), i)
        ki = min(k, elegibles.size)
        salida.append(
            tuple(sorted(rng.choice(elegibles, size=ki, replace=False).tolist()))
        )
    return tuple(salida)


def clausura(padres: tuple[tuple[int, ...], ...], semilla: set[int]) -> set[int]:
    """Cierre de `semilla`: agrega `padres[i]` mientras alguno intersecte parcialmente.

    Al terminar, todo nodo tiene *todos* o *ninguno* de sus padres en el resultado.
    """
    A = set(semilla)
    conjuntos = [set(p) for p in padres]
    creciendo = True
    while creciendo:
        creciendo = False
        for P in conjuntos:
            if P & A and not P <= A:
                A |= P
                creciendo = True
    return A


def corte_cerrado(
    padres: tuple[tuple[int, ...], ...],
) -> tuple[frozenset[int], frozenset[int]] | None:
    """`(A, alcance)` de un corte cerrado **propio**, o `None` si no existe.

    Cualquier corte así da `f = 0`, así que su existencia vuelve trivial a la red.

    Basta probar la clausura de cada dim suelta, y es completo: si existe un cerrado
    propio `A`, tomando `d ∈ A` la clausura de `{d}` nunca sale de `A` —porque `A` ya
    es cerrado— así que también es propia.
    """
    n = len(padres)
    for d in range(n):
        A = clausura(padres, {d})
        if 0 < len(A) < n:
            alcance = frozenset(i for i in range(n) if set(padres[i]) <= A)
            return frozenset(A), alcance
    return None


def es_separable(padres: tuple[tuple[int, ...], ...]) -> bool:
    """`True` si la red admite un corte cerrado propio, o sea si `φ = 0` es forzoso."""
    return corte_cerrado(padres) is not None
