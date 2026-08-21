"""¿Existe una especificación compacta encontrable para esta TPM?

    uv run python -m src.io.identificabilidad N10A N15A N20A

El oráculo por Zeta local (`strategies/python/zeta_local.py`) necesita la spec
compacta —padres y tabla por nodo— y no la TPM materializada. Para una red generada
desde una spec eso es gratis. Para una TPM ya existente hay que **reconstruirla**, y
antes de intentarlo conviene saber si es siquiera posible.

Dos maneras de que no lo sea, y hay que distinguirlas:

1. **Densa.** Cada nodo depende de verdad de todas las dims. Una función booleana
   aleatoria de `n` variables necesita `2^n` bits para describirse: no hay
   compresión posible, por conteo. Ninguna técnica lo evade.
2. **Bajo el ruido.** La TPM es una estimación por muestreo y las dependencias son
   más chicas que su propio error. No se puede distinguir "el nodo `i` depende de la
   dim `d`" de "es ruido de estimación". No es un problema de algoritmo sino de datos.

Este módulo mide las dos cosas y emite un veredicto. Un "no identificable" es un
resultado, no un fallo: dice que hay que ir a buscar la fuente del sistema en vez de
invertir la matriz.
"""

import sys

import numpy as np

#: Una influencia se considera señal si supera este múltiplo del piso de ruido.
FACTOR_SENAL = 2.0


def influencias(tpm: np.ndarray) -> np.ndarray:
    """`(N, D)` con `mean |H_i(·, d=0) − H_i(·, d=1)|`.

    Cuánto cambia la probabilidad del nodo `i` al voltear la dim `d`, promediado
    sobre las demás. Cero exacto significa que `i` no depende de `d`.

    Convención little-endian: en el estado con índice `s`, la dim `d` está encendida
    sii `(s >> d) & 1` — la misma de `NCube.ndata` (reshape con `order="F"`).
    """
    filas, n = tpm.shape
    idx = np.arange(filas, dtype=np.int64)
    salida = np.empty((n, n), dtype=np.float64)
    for d in range(n):
        bajo = idx[((idx >> d) & 1) == 0]
        alto = bajo | (1 << d)
        salida[:, d] = np.abs(tpm[bajo, :] - tpm[alto, :]).mean(axis=0)
    return salida


def piso_de_ruido(tpm: np.ndarray) -> tuple[float, int | None]:
    """`(error esperado entre dos celdas, muestras estimadas por estado)`.

    Si la TPM sólo tiene `{0, 1}` es determinista: no hubo muestreo y el piso es 0.

    Si no, se estima `m` desde los **denominadores** de los valores observados: una
    frecuencia empírica de `m` muestras es un múltiplo de `1/m`. El error estándar
    por celda es `√(p(1−p)/m) ≤ √(0.25/m)`, y dos celdas independientes difieren en
    promedio `≈ 0.8·√2·σ`.

    ⚠️ La estimación de `m` es **cruda**: toma el denominador más chico que explica
    cada valor y se queda con la mediana. Sirve para un orden de magnitud, no más.
    """
    v = np.unique(tpm)
    v = v[(v > 1e-12) & (v < 1 - 1e-12)]
    if v.size == 0:
        return 0.0, None

    dens = []
    for x in v[:200]:
        for d in range(2, 5000):
            if abs(x * d - round(x * d)) < 1e-9:
                dens.append(d)
                break
    if not dens:
        return 0.0, None
    m = int(np.median(dens))
    sigma = np.sqrt(0.25 / m)
    return float(0.8 * np.sqrt(2) * sigma), m


def diagnosticar(tpm: np.ndarray, nombre: str = "") -> dict:
    """Veredicto sobre si una spec compacta es encontrable para esta TPM."""
    infl = influencias(tpm)
    n = tpm.shape[1]
    piso, muestras = piso_de_ruido(tpm)
    umbral = FACTOR_SENAL * piso

    senal = infl > max(umbral, 1e-12)
    fan_in = senal.sum(axis=1)

    if piso > 0 and not senal.any():
        veredicto = "NO IDENTIFICABLE — toda la señal está bajo el ruido de muestreo"
    elif int(fan_in.max()) >= n:
        veredicto = "DENSA — algún nodo depende de las n dims; sin spec compacta"
    else:
        veredicto = f"IDENTIFICABLE — fan-in estimado ≤ {int(fan_in.max())} de {n}"

    return {
        "nombre": nombre,
        "n": n,
        "influencias": infl,
        "piso_ruido": piso,
        "muestras_estimadas": muestras,
        "fan_in_estimado": fan_in,
        "veredicto": veredicto,
    }


def informe(d: dict) -> str:
    infl = d["influencias"]
    piso, m = d["piso_ruido"], d["muestras_estimadas"]
    origen = "determinista (sin muestreo)" if m is None else f"≈{m} muestras/estado"
    ratio = float(np.median(infl)) / piso if piso > 0 else float("inf")
    return (
        f"{d['nombre'] or 'TPM'}  n={d['n']}\n"
        f"  influencia por (nodo,dim): min={infl.min():.4f} "
        f"mediana={np.median(infl):.4f} max={infl.max():.4f}\n"
        f"  piso de ruido: {piso:.4f}  [{origen}]"
        + (f"   ratio mediana/ruido = {ratio:.2f}\n" if piso > 0 else "\n")
        + f"  fan-in estimado por nodo: min={int(d['fan_in_estimado'].min())} "
        f"max={int(d['fan_in_estimado'].max())}\n"
        f"  -> {d['veredicto']}"
    )


def main(nombres: list[str]) -> None:
    from src.io.manager import cargar_mpt

    for nombre in nombres:
        print(informe(diagnosticar(np.asarray(cargar_mpt(nombre)), nombre)))
        print()


if __name__ == "__main__":
    main(sys.argv[1:] or ["N10A", "N15A", "N20A"])
