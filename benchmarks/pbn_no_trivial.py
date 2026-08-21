"""¿El régimen disperso deja de ser trivial con nodos probabilísticos?

    uv run python benchmarks/pbn_no_trivial.py [n] [n_cruce]

Motivo. Con nodos **binarios**, `mean_A(δᵢ) = mean_A(Hᵢ) − pᵢ` y `Hᵢ, pᵢ ∈ {0,1}`,
así que ese término vale 0 en cuanto la función del nodo es **constante sobre la
cara**. Con `k` chico las caras son chicas y eso pasa seguido: `φ = 0` desde n=20 con
k=3, o sea justo en el régimen donde el oráculo local gana. Con probabilidades en
`(0,1)` la constancia exacta es de medida cero.

Dos bloques:

1. `φ` mínimo por (k, modelo) sobre el oráculo local. El binario es el control y
   debe dar 0; los PBN deben dar `φ > 0` con k chico.
2. Cruce contra `analytic` sobre la TPM materializada de la MISMA red, a un `n`
   donde ambas rutas corren, para confirmar que siguen coincidiendo con tablas no
   binarias.
"""

import sys
import time

import numpy as np

from src.iit.core.params import Params
from src.iit.strategies.python.qsw.core import stoer_wagner_queyranne
from src.iit.strategies.python.zeta_local import (
    f_cara_batch_local,
    preparar_oraculo_local,
)
from src.iit.strategies.runner import ejecutar
from src.io.fst import generar_espec, generar_espec_pbn, tpm_nodo_estado

MODELOS = {
    "binario": lambda n, k, s: generar_espec(n, k, semilla=s),
    "ruido=0.05": lambda n, k, s: generar_espec_pbn(n, k, ruido=0.05, semilla=s),
    "ruido=0.20": lambda n, k, s: generar_espec_pbn(n, k, ruido=0.20, semilla=s),
    "PBN L=3": lambda n, k, s: generar_espec_pbn(n, k, funciones=3, semilla=s),
}


def _f_batch(espec, estado):
    """Oráculo local envuelto para `stoer_wagner_queyranne`. Nunca materializa la TPM."""
    orc = preparar_oraculo_local(espec, estado)
    D = N = espec.n
    desp = np.arange(N, dtype=np.int64)
    mec_mask = (1 << D) - 1

    def f_batch(masks):
        # Partir en Python ANTES de bajar a int64: con V = 2n grande la máscara
        # completa no entra en un C long, cada mitad sí.
        mec = np.fromiter(
            (int(x) & mec_mask for x in masks), dtype=np.int64, count=len(masks)
        )
        alt = np.fromiter(
            (int(x) >> D for x in masks), dtype=np.int64, count=len(masks)
        )
        return f_cara_batch_local(orc, mec, ((alt[:, None] >> desp) & 1).astype(bool))

    return f_batch, orc


def minimo(espec, estado):
    valor, _, _ = stoer_wagner_queyranne(
        2 * espec.n, _f_batch(espec, estado)[0], modo="estatico"
    )
    return valor


def bloque_trivialidad(n: int, semillas=(5, 11, 23)):
    print(f"φ mínimo sobre el oráculo local — n={n}, mediana de {len(semillas)} semillas\n")
    print(f"  {'modelo':<12} " + "".join(f"k={k:<10}" for k in (2, 3, 5)))
    for nombre, hacer in MODELOS.items():
        fila = []
        for k in (2, 3, 5):
            vals = []
            for s in semillas:
                espec = hacer(n, k, s)
                vals.append(minimo(espec, tuple(0 for _ in range(n))))
            fila.append(float(np.median(vals)))
        marca = "  <- trivial" if all(v == 0.0 for v in fila) else ""
        print(f"  {nombre:<12} " + "".join(f"{v:<12.6f}" for v in fila) + marca)


def bloque_cruce(n: int):
    print(f"\nCruce contra `analytic` sobre la TPM materializada — n={n}\n")
    print(f"  {'modelo':<12} {'k':<3} {'estado':<8} {'local':<12} {'analytic':<12} ok")
    for nombre, hacer in MODELOS.items():
        for k in (2, 3):
            espec = hacer(n, k, 5)
            tpm = tpm_nodo_estado(espec)
            for s in (0, (1 << n) - 1):
                estado = tuple((s >> j) & 1 for j in range(n))
                txt = "".join(map(str, estado))
                loc = minimo(espec, estado)
                ana = ejecutar(tpm, Params(txt, "1" * n, "1" * n, "1" * n), "analytic").perdida
                ok = abs(loc - ana) < 1e-5
                print(f"  {nombre:<12} {k:<3} {txt:<8} {loc:<12.6f} {ana:<12.6f} "
                      f"{'ok' if ok else '<-- DIFIERE'}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    n_cruce = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.perf_counter()
    bloque_trivialidad(n)
    bloque_cruce(n_cruce)
    print(f"\n({time.perf_counter() - t0:.1f}s)")
