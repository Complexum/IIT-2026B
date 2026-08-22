"""Fórmulas booleanas en CNF (forma normal conjuntiva), convención DIMACS.

Es el formato de **autoría y persistencia** de las funciones de nodo. La tabla de
verdad es el compilado interno: se deriva al cargar y no se guarda, porque
`[0,1,1,1]` no se lee y `[[2,3]]` sí.

Convención
----------
- Variable `v > 0` ↔ nodo `v - 1`. Los nodos se numeran desde 1 hacia afuera y
  desde 0 hacia adentro, igual que DIMACS.
- Literal negativo `-v` ↔ el nodo `v - 1` negado.
- Una **cláusula** es una lista de literales unidos por **OR**.
- La **fórmula** es la lista de cláusulas unidas por **AND**.

Ejemplo — la red `A(B,C)=B+C ; B(A,C)=A·C ; C(A,B)=A xor B` con A=1, B=2, C=3::

    {"1": [[2, 3]],            # A = B ∨ C
     "2": [[1], [3]],          # B = A ∧ C
     "3": [[1, 2], [-1, -2]]}  # C = (A ∨ B) ∧ (¬A ∨ ¬B) = A xor B

Casos límite, con la semántica estándar: `[]` (cero cláusulas) es el AND vacío,
o sea la constante **1**; `[[]]` contiene la cláusula vacía, que es el OR vacío,
o sea la constante **0**.

Sobre el tamaño
---------------
CNF es completa —toda función booleana se puede escribir así— pero no siempre es
compacta: la CNF canónica de una tabla aleatoria de `2^k` tiene ~`2^(k-1)`
cláusulas de `k` literales, o sea ~`k/2` veces más números que la tabla. Es el
formato correcto para redes **escritas a mano** y para funciones estructuradas
(and / or / mayoría / umbral), no para tablas aleatorias densas.
"""

import numpy as np

#: Una cláusula es una lista de literales; una fórmula, una lista de cláusulas.
Clausula = list[int]
FormulaCNF = list[Clausula]


def validar(clausulas: FormulaCNF, n: int) -> None:
    """Lanza si algún literal es 0 o nombra una variable fuera de `1..n`."""
    for c, clausula in enumerate(clausulas):
        for lit in clausula:
            if lit == 0:
                raise ValueError(f"cláusula {c}: el literal 0 no es válido")
            if not (1 <= abs(lit) <= n):
                raise ValueError(
                    f"cláusula {c}: literal {lit} fuera de rango (variables 1..{n})"
                )


def variables_de(clausulas: FormulaCNF) -> tuple[int, ...]:
    """Nodos (0-indexados) que la fórmula menciona — o sea, **los padres**.

    Se infieren de la fórmula en vez de declararse aparte: así no pueden quedar
    desincronizados con ella.
    """
    return tuple(sorted({abs(lit) - 1 for clausula in clausulas for lit in clausula}))


def evaluar(clausulas: FormulaCNF, padres: tuple[int, ...]) -> np.ndarray:
    """Compila la fórmula a la tabla de verdad sobre `padres`. Devuelve `2^k` floats.

    El bit `j` del índice de la tabla es `padres[j]` — little-endian, la misma
    convención que el índice de estado del pipeline.
    """
    k = len(padres)
    pos = {p: j for j, p in enumerate(padres)}
    u = np.arange(1 << k, dtype=np.int64)

    valor = np.ones(1 << k, dtype=bool)  # AND vacío = 1
    for clausula in clausulas:
        acum = np.zeros(1 << k, dtype=bool)  # OR vacío = 0
        for lit in clausula:
            bit = ((u >> pos[abs(lit) - 1]) & 1).astype(bool)
            acum |= bit if lit > 0 else ~bit
        valor &= acum
    return valor.astype(np.float64)


def depende_de_todas(tabla: np.ndarray, k: int) -> bool:
    """¿La tabla depende de verdad de sus `k` entradas?

    Una tabla aleatoria puede salir degenerada: con k=2, 2 de las 16 posibles son
    constantes y 8 más dependen de una sola entrada. Si se admiten, el fan-in
    efectivo es menor que el pedido y las mediciones que lo usan como perilla quedan
    sucias. Vale igual para tablas de probabilidades, no sólo binarias.
    """
    nd = tabla.reshape((2,) * k, order="F")
    return all(
        not np.array_equal(np.take(nd, 0, axis=j), np.take(nd, 1, axis=j))
        for j in range(k)
    )


def condicionar(clausulas: FormulaCNF, variable: int, valor: int) -> FormulaCNF:
    """Fija `variable` (1-indexada) en `valor` — **propagación unitaria**.

    Es el paso clásico de Davis-Putnam: una cláusula que contiene el literal
    satisfecho se elimina entera (ya es verdadera), y de las demás se borra el
    literal falsificado. CNF es **cerrada** bajo esta operación.

    Casos límite: si queda una cláusula vacía la fórmula es la constante 0; si no
    queda ninguna cláusula, la constante 1.
    """
    satisfecho = variable if valor else -variable
    falsificado = -satisfecho
    salida: FormulaCNF = []
    for clausula in clausulas:
        if satisfecho in clausula:
            continue  # la cláusula ya es verdadera
        salida.append([lit for lit in clausula if lit != falsificado])
    return salida


def desde_tabla(tabla: np.ndarray, padres: tuple[int, ...]) -> FormulaCNF:
    """CNF canónica de una tabla de verdad: una cláusula por asignación falsa.

    Cada cláusula excluye exactamente un patrón donde la función vale 0, así que
    el AND de todas es la función. Es la inversa de `evaluar`, y sirve para que una
    red generada al azar se pueda guardar en el mismo formato que una escrita a mano.
    """
    k = len(padres)
    if not np.isin(tabla, (0.0, 1.0)).all():
        raise ValueError("CNF sólo representa funciones binarias")
    if tabla.all():
        return []  # constante 1
    if not tabla.any():
        return [[]]  # constante 0

    clausulas: FormulaCNF = []
    for m in np.flatnonzero(tabla == 0.0):
        # Literal falso en el patrón m: positivo si el bit está en 0, negativo si en 1.
        clausulas.append(
            [
                -(padres[j] + 1) if (int(m) >> j) & 1 else (padres[j] + 1)
                for j in range(k)
            ]
        )
    return clausulas


def nombre_de_variable(indice: int) -> str:
    """Índice 0-based → nombre estilo columna de hoja de cálculo: `A…Z, AA, AB, …`.

    El alfabeto se agota a 26 nodos y el oráculo por Zeta local llega a 60+, así que
    hace falta un esquema ilimitado. Base-26 tiene una propiedad que lo hace natural
    acá: las variables del `.spec.json` son **1-indexadas** (DIMACS) y las columnas
    de hoja de cálculo también, así que el nombre *es* el número de variable escrito
    en base 26 — `A=1`, `Z=26`, `AA=27`.

    Descartado el alfabeto griego: en este proyecto `δ`, `Σ`, `φ`, `Θ`, `σ` y `ε` ya
    tienen significado matemático (`δ = H − p`, `φ` la cantidad calculada), y encima
    trae colisiones visuales propias (`ο`/`o`/`0`, `ι`/`i`, `ν`/`v`, `ρ`/`p`).
    """
    v = indice + 1
    nombre = ""
    while v:
        v, resto = divmod(v - 1, 26)
        nombre = chr(ord("A") + resto) + nombre
    return nombre


def nombres_de_variables(n: int) -> list[str]:
    """Los `n` primeros nombres de variable, en orden."""
    return [nombre_de_variable(i) for i in range(n)]


def formatear(clausulas: FormulaCNF, nombres: list[str] | None = None) -> str:
    """Fórmula legible, para mensajes de error y depuración."""
    if not clausulas:
        return "1"
    if any(not c for c in clausulas):
        return "0"

    def lit(v: int) -> str:
        nom = nombres[abs(v) - 1] if nombres else str(abs(v))
        return f"¬{nom}" if v < 0 else nom

    return "".join(
        ("⟨" + " + ".join(lit(v) for v in c) + "⟩") if len(c) > 1 else lit(c[0])
        for c in clausulas
    )
