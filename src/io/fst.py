from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.io import cnf
from src.iit.base.app import aplicacion
from src.iit.base.consts import CSV_EXTENSION

REDES_DIR = Path("data/input/networks")

_optimizar_executor = ThreadPoolExecutor(max_workers=1)

# ─────────────────────────────────────────────────────────────────────────────
# FST booleano: la red se define por su ESTRUCTURA, no celda por celda.
# ─────────────────────────────────────────────────────────────────────────────
#
# `generator.generar_red` sortea cada celda de la TPM de forma independiente. Eso
# NO es un FST: no hay aristas ni función por nodo, y el resultado es una función
# booleana aleatoria densa — cada nodo depende de las n dimensiones con influencia
# exactamente 1/2. Medido sobre N20A: 400/400 pares (nodo,dim) con influencia en
# [0.498, 0.502]. Es el peor caso para cualquier método que explote estructura, y
# explica por qué la TPM materializada no se puede comprimir.
#
# Acá la red se define como el sistema original: cada nodo tiene `k` padres y una
# función sobre ellos. La TPM nodo-estado se DERIVA de esa especificación, así que
# el pipeline tradicional la consume sin cambios y sirve de validación.
#
# Convención little-endian, la misma que `zeta.pivote_plano` y `NCube.ndata`
# (order="F"): en el estado con índice plano `s`, el nodo `j` está encendido sii
# `(s >> j) & 1`. Igual dentro de la tabla de un nodo: el bit `j` del índice de
# `tablas[i]` es el padre `padres[i][j]`.


@dataclass(frozen=True)
class EspecFST:
    """Especificación compacta de la red. Es la fuente; la TPM es un derivado.

    `tablas[i]` tiene `2^len(padres[i])` entradas float en [0,1]:
    `tablas[i][m]` = P(nodo i = 1 en t+1 | sus padres tienen el patrón m en t).

    Binario = todas las entradas en {0.0, 1.0}. Lo continuo (ruido de salida,
    frecuencias empíricas, compuertas con pesos) usa exactamente la misma
    estructura con otros valores, así que no requiere otro formato.
    """

    n: int
    padres: tuple[tuple[int, ...], ...]
    tablas: tuple[np.ndarray, ...]
    #: Fórmula CNF por nodo, cuando la red vino de una. Es la FUENTE: `tablas` es
    #: su compilado. Se conserva para que guardar → cargar → guardar devuelva lo
    #: que el autor escribió y no la CNF canónica, que es equivalente pero verbosa
    #: (`A ∧ C` vuelve como `(A∨C) ∧ (¬A∨C) ∧ (A∨¬C)`). `None` en redes generadas.
    formulas: tuple[cnf.FormulaCNF, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.padres) != self.n or len(self.tablas) != self.n:
            raise ValueError("padres y tablas deben tener n entradas")
        for i, (ps, t) in enumerate(zip(self.padres, self.tablas)):
            if len(t) != (1 << len(ps)):
                raise ValueError(f"nodo {i}: tabla de {len(t)} para {len(ps)} padres")
            if any(not (0 <= p < self.n) for p in ps):
                raise ValueError(f"nodo {i}: padre fuera de rango en {ps}")
        if self.formulas is not None and len(self.formulas) != self.n:
            raise ValueError("formulas debe tener n entradas")

    @property
    def fan_in(self) -> tuple[int, ...]:
        return tuple(len(p) for p in self.padres)

    def es_binaria(self) -> bool:
        return all(np.isin(t, (0.0, 1.0)).all() for t in self.tablas)


def _patron_de_padres(estados: np.ndarray, padres: tuple[int, ...]) -> np.ndarray:
    """Índice dentro de la tabla del nodo, para cada estado global. Little-endian."""
    patron = np.zeros(estados.shape, dtype=np.int64)
    for j, p in enumerate(padres):
        patron |= ((estados >> p) & 1) << j
    return patron


def tpm_nodo_estado(espec: EspecFST) -> np.ndarray:
    """Materializa la TPM nodo-estado `(2^n, n)` — el formato que carga `cargar_mpt`.

    Esta es la expansión que crea el `2^n`: un nodo con `k` padres tiene una tabla
    de `2^k`, y acá se replica `2^(n-k)` veces. A n=20 con k=5 son 32768 copias
    idénticas por nodo.
    """
    estados = np.arange(1 << espec.n, dtype=np.int64)
    tpm = np.empty((1 << espec.n, espec.n), dtype=np.float64)
    for i in range(espec.n):
        tpm[:, i] = espec.tablas[i][_patron_de_padres(estados, espec.padres[i])]
    return tpm


def tpm_estado_estado(espec: EspecFST) -> np.ndarray:
    """TPM estado-estado `(2^n, 2^n)` — sólo para validar, es O(4^n).

    Bajo independencia condicional entre nodos dado el estado en t:
    `P(s' | s) = Π_i P(nodo i = s'_i | s)`. Es la matriz que el sistema original
    define; la nodo-estado es su marginalización por nodo.
    """
    tpm = tpm_nodo_estado(espec)
    n, m = espec.n, 1 << espec.n
    destinos = np.arange(m, dtype=np.int64)
    P = np.ones((m, m), dtype=np.float64)
    for i in range(n):
        bit = ((destinos >> i) & 1).astype(np.float64)  # (m,)
        p_on = tpm[:, i][:, None]  # (m,1)
        P *= np.where(bit[None, :] == 1.0, p_on, 1.0 - p_on)
    return P


def marginalizar_a_nodo_estado(P: np.ndarray) -> np.ndarray:
    """Estado-estado → nodo-estado: P(nodo i = 1 | s) = Σ_{s': s'_i = 1} P(s'|s).

    Es la reducción que el pipeline asume. Recuperar exactamente `tpm_nodo_estado`
    a partir de `tpm_estado_estado` prueba que la simplificación no pierde nada
    cuando los nodos son condicionalmente independientes dado el estado en t.
    """
    m = P.shape[1]
    n = int(m).bit_length() - 1
    destinos = np.arange(m, dtype=np.int64)
    return np.stack(
        [P[:, ((destinos >> i) & 1) == 1].sum(axis=1) for i in range(n)], axis=1
    )


# ── construcción y persistencia ──────────────────────────────────────────────

#: Compuertas con nombre. Reciben el patrón de padres como array de bits `(2^k, k)`.
COMPUERTAS = {
    "and": lambda b: b.all(axis=1),
    "or": lambda b: b.any(axis=1),
    "xor": lambda b: b.sum(axis=1) % 2 == 1,
    "mayoria": lambda b: b.sum(axis=1) * 2 > b.shape[1],
}


def _depende_de_todos(tabla: np.ndarray, ki: int) -> bool:
    """¿La tabla depende de verdad de sus `ki` entradas?

    Una tabla de verdad aleatoria puede salir degenerada: con k=2, 2 de las 16
    posibles son constantes y 8 más dependen de una sola entrada. Si se admiten,
    el fan-in efectivo es menor que `k` y las mediciones que usan `k` como perilla
    quedan sucias.
    """
    nd = tabla.reshape((2,) * ki, order="F")
    return all(
        not np.array_equal(np.take(nd, 0, axis=j), np.take(nd, 1, axis=j))
        for j in range(ki)
    )


def generar_espec(
    n: int,
    k: int,
    *,
    compuerta: str | None = None,
    auto_lazo: bool = True,
    fan_in_estricto: bool = True,
    semilla: int | None = None,
) -> EspecFST:
    """FST binario con `k` padres por nodo.

    Args:
        n: nodos.
        k: fan-in por nodo (se recorta a `n`). **Es la perilla que importa**: con
            `k = n` se recupera el peor caso denso que produce `generar_red`.
        compuerta: `None` → tabla de verdad aleatoria sobre los `k` padres;
            si no, una de `COMPUERTAS` aplicada a los padres.
        fan_in_estricto: re-sortea la tabla hasta que dependa de sus `k` padres,
            para que el fan-in efectivo sea el pedido y no menos.
        auto_lazo: si el nodo puede ser padre de sí mismo (el estado de un nodo en
            t+1 depende del suyo en t — semántica normal de un FST).
        semilla: si es None usa `aplicacion.semilla_numpy`.
    """
    if n < 1:
        raise ValueError("n debe ser positivo")
    k = max(1, min(k, n))
    rng = np.random.default_rng(
        aplicacion.semilla_numpy if semilla is None else semilla
    )

    padres, tablas = [], []
    for i in range(n):
        elegibles = np.arange(n) if auto_lazo else np.delete(np.arange(n), i)
        ki = min(k, elegibles.size)
        ps = tuple(sorted(rng.choice(elegibles, size=ki, replace=False).tolist()))
        if compuerta is None:
            t = rng.integers(2, size=1 << ki).astype(np.float64)
            for _ in range(256):
                if not fan_in_estricto or _depende_de_todos(t, ki):
                    break
                t = rng.integers(2, size=1 << ki).astype(np.float64)
            else:
                raise RuntimeError(
                    f"nodo {i}: no se logró una tabla no degenerada con k={ki}"
                )
        else:
            fn = COMPUERTAS.get(compuerta)
            if fn is None:
                raise ValueError(
                    f"compuerta desconocida: {compuerta!r} (usar {sorted(COMPUERTAS)})"
                )
            u = np.arange(1 << ki, dtype=np.int64)
            bits = ((u[:, None] >> np.arange(ki)) & 1).astype(bool)
            t = fn(bits).astype(np.float64)
        padres.append(ps)
        tablas.append(t)
    return EspecFST(n=n, padres=tuple(padres), tablas=tuple(tablas))


def _ruta_spec(nombre: str) -> Path:
    return REDES_DIR / f"{nombre}.spec.json"


def espec_desde_cnf(formulas: dict[int, cnf.FormulaCNF]) -> EspecFST:
    """Compila `{variable: cláusulas}` a `EspecFST`.

    `n` sale de la cantidad de nodos y los padres de las variables que cada
    fórmula menciona: nada se declara dos veces, así que nada puede quedar
    desincronizado. Las claves deben ser exactamente `1..n`.
    """
    n = len(formulas)
    faltan = set(range(1, n + 1)) - set(formulas)
    if faltan:
        raise ValueError(f"faltan los nodos {sorted(faltan)} (se esperan 1..{n})")

    padres, tablas = [], []
    for v in range(1, n + 1):
        clausulas = formulas[v]
        cnf.validar(clausulas, n)
        ps = cnf.variables_de(clausulas)
        padres.append(ps)
        tablas.append(cnf.evaluar(clausulas, ps))
    return EspecFST(
        n=n,
        padres=tuple(padres),
        tablas=tuple(tablas),
        formulas=tuple(formulas[v] for v in range(1, n + 1)),
    )


def cnf_de_espec(espec: EspecFST) -> dict[int, cnf.FormulaCNF]:
    """`EspecFST` → `{variable: cláusulas}`. Inversa de `espec_desde_cnf`.

    Devuelve la fórmula original si la red vino de una; si no (red generada al
    azar), la CNF canónica derivada de la tabla.
    """
    if espec.formulas is not None:
        return {i + 1: espec.formulas[i] for i in range(espec.n)}
    return {
        i + 1: cnf.desde_tabla(espec.tablas[i], espec.padres[i]) for i in range(espec.n)
    }


def guardar_espec(nombre: str, espec: EspecFST) -> Path:
    """Sidecar JSON junto al CSV. Su existencia marca a la red como estructurada.

    Guarda **sólo las fórmulas CNF**, una por nodo. La tabla de verdad es un
    compilado y no se persiste; `n`, los padres y la matriz de adyacencia se
    derivan al cargar.
    """
    import json

    ruta = _ruta_spec(nombre)
    formulas = cnf_de_espec(espec)
    cuerpo = ",\n".join(
        f' "{v}": {json.dumps(formulas[v])}' for v in sorted(formulas)
    )
    ruta.write_text("{\n" + cuerpo + "\n}\n")
    return ruta


def cargar_espec(nombre: str) -> EspecFST | None:
    """Especificación compacta de la red, o None si la red no la tiene."""
    import json

    ruta = _ruta_spec(nombre)
    if not ruta.exists():
        return None
    crudo = json.loads(ruta.read_text())
    return espec_desde_cnf({int(k): v for k, v in crudo.items()})


def generar_red_booleana(
    n: int,
    k: int,
    *,
    compuerta: str | None = None,
    auto_lazo: bool = True,
    semilla: int | None = None,
    optimizar: bool = True,
) -> tuple[str, EspecFST]:
    """Genera un FST binario y escribe su TPM nodo-estado + la especificación.

    El CSV es idéntico en formato al de `generar_red`, así que `cargar_mpt` y todo
    el pipeline lo consumen sin cambios: eso es lo que permite validar la ruta
    estructurada contra la tradicional sobre la misma red.

    Returns:
        `(nombre_csv, espec)`.
    """
    espec = generar_espec(
        n, k, compuerta=compuerta, auto_lazo=auto_lazo, semilla=semilla
    )

    REDES_DIR.mkdir(parents=True, exist_ok=True)
    sufijo = "A"
    while (REDES_DIR / f"N{n}{sufijo}.{CSV_EXTENSION}").exists():
        sufijo = chr(ord(sufijo) + 1)
    nombre_base = f"N{n}{sufijo}"

    tpm = tpm_nodo_estado(espec)
    es_bin = espec.es_binaria()
    np.savetxt(
        REDES_DIR / f"{nombre_base}.{CSV_EXTENSION}",
        tpm,
        delimiter=",",
        fmt="%d" if es_bin else "%.6f",
    )
    guardar_espec(nombre_base, espec)

    if optimizar:
        from src.io.manager import optimizar_red

        _optimizar_executor.submit(optimizar_red, nombre_base)

    return f"{nombre_base}.{CSV_EXTENSION}", espec
