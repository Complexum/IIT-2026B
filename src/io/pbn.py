"""Redes booleanas probabilísticas (PBN).

Modelo de Shmulevich et al., *Bioinformatics* 18(2):261 (2002): cada nodo tiene un
**conjunto** de funciones booleanas y en cada paso elige una al azar con una
probabilidad de selección. Es decir, "a veces la operación da 1 y a veces 0", que es
lo que hace falta para que una red tenga probabilidades y no sólo `{0, 1}`.

Por qué existe
--------------
El régimen disperso **binario** es trivial: `mean_A(δᵢ) = mean_A(Hᵢ) − pᵢ`, y con
`Hᵢ, pᵢ ∈ {0,1}` ese término es 0 en cuanto la función del nodo es **constante sobre
la cara**. Con `k` chico las caras son chicas y eso pasa seguido, así que `φ = 0`
desde n=20 con k=3. Con probabilidades en `(0,1)` la constancia exacta es de medida
cero y el régimen deja de ser trivial.

Este módulo sólo sabe de **fórmulas de un nodo**. No conoce `EspecFST` ni las TPMs:
`fst.py` compone la topología con lo de acá. Así no hay ciclo de importación.
"""

import numpy as np

from src.io import cnf

#: Especificación de un nodo: o una fórmula CNF (determinista), o una lista de
#: pares `(probabilidad, fórmula)` — un PBN.
EspecNodo = cnf.FormulaCNF | list


def es_pbn(spec) -> bool:
    """Discrimina un nodo PBN de uno determinista.

    Una **cláusula** es una lista plana de enteros; una **entrada PBN** es un par
    `[probabilidad, lista_de_cláusulas]`, cuyo segundo elemento es una lista::

        [[2, 3]]                          → determinista: la cláusula (B ∨ C)
        [[1], [3]]                        → determinista: A ∧ C
        [[0.7, [[1], [3]]], [0.3, [[2]]]] → PBN: pares (prob, fórmula)

    Es el único punto ambiguo del formato, así que la regla vive acá y está testeada.
    Los casos límite `[]` (constante 1) y `[[]]` (constante 0) son deterministas.
    """
    return (
        bool(spec)
        and isinstance(spec[0], (list, tuple))
        and len(spec[0]) == 2
        and isinstance(spec[0][1], (list, tuple))
    )


def compilar(spec: EspecNodo, n: int, v: int) -> tuple[tuple[int, ...], np.ndarray]:
    """`(padres, tabla)` de un nodo PBN.

    En cada paso se elige una fórmula al azar, así que la probabilidad de encendido
    es la mezcla convexa::

        tabla[m] = Σ_l  c_l · f_l(m)

    Los padres son la **unión** de las variables de todas sus fórmulas, y cada
    fórmula se evalúa sobre esa unión (`cnf.evaluar` acepta un superconjunto de sus
    propias variables).
    """
    pares = [(float(c), list(f)) for c, f in spec]
    if any(c < 0.0 for c, _ in pares):
        raise ValueError(f"nodo {v}: probabilidad de selección negativa")
    total = sum(c for c, _ in pares)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"nodo {v}: las probabilidades suman {total}, deben sumar 1")

    for _, f in pares:
        cnf.validar(f, n)
    padres = tuple(sorted({x for _, f in pares for x in cnf.variables_de(f)}))

    tabla = np.zeros(1 << len(padres), dtype=np.float64)
    for c, f in pares:
        tabla += c * cnf.evaluar(f, padres)
    return padres, tabla


def generar_formulas(
    padres: tuple[tuple[int, ...], ...],
    rng: np.random.Generator,
    *,
    funciones: int = 2,
    ruido: float | None = None,
    fan_in_estricto: bool = True,
) -> dict[int, EspecNodo]:
    """Especificación PBN de cada nodo, dada la topología ya elegida.

    Recibe `padres` en vez de elegirlos para no duplicar la selección de topología,
    que vive en `fst.elegir_padres`.

    Args:
        funciones: cuántas fórmulas por nodo, con probabilidades de selección
            aleatorias que suman 1.
        ruido: atajo para el caso de un solo mecanismo con volteo de salida,
            `P = ε + (1−2ε)·f`, expresado como el PBN `[(1−ε, f), (ε, ¬f)]`. La
            negación sale de la tabla complementaria. Excluye a `funciones`.
    """
    if ruido is not None and not 0.0 <= ruido <= 0.5:
        raise ValueError(f"ruido debe estar en [0, 0.5], no {ruido}")
    if ruido is None and funciones < 1:
        raise ValueError("funciones debe ser >= 1")

    formulas: dict[int, EspecNodo] = {}
    for i, ps in enumerate(padres):
        k = len(ps)
        if ruido is not None:
            pesos = np.array([1.0 - ruido, ruido])
            armar = lambda: (lambda b: [b, 1.0 - b])(
                rng.integers(2, size=1 << k).astype(np.float64)
            )
        else:
            pesos = rng.random(funciones)
            pesos /= pesos.sum()
            armar = lambda: [
                rng.integers(2, size=1 << k).astype(np.float64)
                for _ in range(funciones)
            ]

        # La mezcla `Σ c_l f_l` es lo que ve el oráculo, así que es ELLA la que tiene
        # que depender de los `k` padres. Con tablas aleatorias sueltas es fácil que
        # salga constante y el nodo quede con fan-in 0.
        for _ in range(256):
            tablas = armar()
            if fan_in_estricto and not cnf.depende_de_todas(
                sum(c * x for c, x in zip(pesos, tablas)), k
            ):
                continue
            break
        else:
            raise RuntimeError(
                f"nodo {i + 1}: no se logró una mezcla no degenerada con k={k}"
            )

        formulas[i + 1] = [
            [float(c), cnf.desde_tabla(x, ps)] for c, x in zip(pesos, tablas)
        ]
    return formulas
