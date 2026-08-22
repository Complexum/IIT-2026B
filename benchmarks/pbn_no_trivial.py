"""¿Cuándo deja de ser trivial el régimen disperso?

    uv run python benchmarks/pbn_no_trivial.py [n] [n_cruce]

El régimen donde el oráculo local gana (`k` chico) daba `φ = 0` siempre. Hay **tres**
mecanismos que producen ese cero, y hacen falta las tres contramedidas juntas:

1. **Corte cerrado.** Si `A ∩ padres(i) = ∅` la cara colapsa al pivote, donde
   `δᵢ = 0` por construcción. Basta un `A` donde cada nodo tenga todos o ninguno de
   sus padres. Es topológico: ni las probabilidades ni el ruido lo tocan.
   → contramedida: `topologia="anillo"` (ver `io/topologia.py`).

2. **Corte de una sola dim con alcance vacío.** Con `A = {d}` y alcance `∅`,
   `mean_{d}(δᵢ) = (δᵢ(pivote) + δᵢ(pivote ⊕ eᵈ))/2` y el primer término es 0. Con
   tablas **binarias** el segundo es 0 o ±½, así que se anula exacto con probabilidad
   ~¼ por hijo — y la búsqueda prueba las `D` dims. Con tablas probabilísticas es un
   real genérico y la cancelación exacta es de medida cero.
   → contramedida: PBN.

3. **El estado pivote.** Medir sólo en el estado todo-ceros enmascara los otros dos:
   ahí muchos nodos son insensibles a un volteo simple.
   → contramedida: barrer estados iniciales.

Dos bloques:

1. Fracción de estados con `φ = 0` por (modelo, topología, k).
2. Cruce contra `analytic` sobre la TPM materializada de la MISMA red, a un `n` donde
   ambas rutas corren, para confirmar que siguen coincidiendo con tablas no binarias.
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
from src.io.topologia import es_separable

MODELOS = {
    "binario": lambda n, k, s, t: generar_espec(n, k, topologia=t, semilla=s),
    "ruido=0.20": lambda n, k, s, t: generar_espec_pbn(n, k, ruido=0.20, topologia=t, semilla=s),
    "PBN L=3": lambda n, k, s, t: generar_espec_pbn(n, k, funciones=3, topologia=t, semilla=s),
}
TOPOLOGIAS = ("aleatoria", "anillo")


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


def bloque_trivialidad(n: int, estados: int = 24, semilla: int = 5):
    print(f"Fracción de estados iniciales con φ = 0 — n={n}, {estados} estados al azar\n")
    print(f"  {'modelo':<12}{'topología':<12}{'k':<3}{'sep':<5}{'ceros':<9}{'mediana φ':<12}max φ")
    rng = np.random.default_rng(0)
    muestras = [
        tuple(int(b) for b in rng.integers(0, 2, size=n)) for _ in range(estados)
    ]
    for nombre, hacer in MODELOS.items():
        for topo in TOPOLOGIAS:
            for k in (2, 3):
                espec = hacer(n, k, semilla, topo)
                sep = "sí" if es_separable(espec.padres) else "no"
                v = np.array([minimo(espec, e) for e in muestras])
                marca = "  <- no trivial" if (v > 0).all() else ""
                print(
                    f"  {nombre:<12}{topo:<12}{k:<3}{sep:<5}"
                    f"{f'{(v == 0).sum()}/{estados}':<9}"
                    f"{np.median(v):<12.6f}{v.max():.6f}{marca}"
                )


def bloque_cruce(n: int):
    print(f"\nCruce contra `analytic` sobre la TPM materializada — n={n}\n")
    print(f"  {'modelo':<12} {'k':<3} {'estado':<8} {'local':<12} {'analytic':<12} ok")
    for nombre, hacer in MODELOS.items():
        for k in (2, 3):
            espec = hacer(n, k, 5, "anillo")
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
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    n_cruce = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    t0 = time.perf_counter()
    bloque_trivialidad(n)
    bloque_cruce(n_cruce)
    print(f"\n({time.perf_counter() - t0:.1f}s)")
