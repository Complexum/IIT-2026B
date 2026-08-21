"""Escalado fuerte del Zeta paralelo: ¿cuánto compra repartir filas entre procesos?

    uv run python benchmarks/escalado_mul.py [n_min] [n_max] [repeticiones]

Existe para responder una pregunta abierta y no para adornar. `QSW.md` §6 cierra
diciendo que, por estar el Zeta limitado por ancho de banda, más hilos en la misma
máquina compran poco porque comparten la DRAM. Es una hipótesis razonable pero no
verificada: los 17-19 GB/s medidos son ~7 % de lo que da un M4 Pro (~273 GB/s) y
menos todavía de un EPYC 7302 (8 canales DDR4-3200, ~200 GB/s agregados), o sea que
**un solo core está lejos de saturar**. Si la hipótesis es correcta la curva se
aplana temprano; si no, escala hasta saturar el bus. Las dos respuestas sirven.

Se cronometra `preparar_oraculo` contra `preparar_oraculo_mul` —el Zeta desnudo— y
no la estrategia entera, porque la búsqueda son 2.8 ms y metería ruido constante
que comprime todos los speedups hacia 1 (el mismo error que `preparar_subsistema`
escondiendo la ventaja de `backend=c`, ver `runner.preparar_subsistema`).

Se reporta el **mínimo** de las repeticiones, no la media: el mínimo es la
estimación menos contaminada por otros procesos de la máquina.
"""

import os
import sys
import time

import numpy as np

from src.iit.core.params import Params
from src.iit.strategies.python.qn.oracle import preparar_oraculo
from src.iit.strategies.python.qsw_mul.zeta_mul import preparar_oraculo_mul
from src.io.manager import reducir_a_subsistema

WORKERS = [2, 4, 8, 12, 16]


def _sub(n, rng):
    """TPM sintética → subsistema. Mismo helper que `benchmarks/calibrar_k.py`."""
    tpm = rng.random((1 << n, n))
    return reducir_a_subsistema(
        tpm, Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)
    )


def _cronometrar(fn, reps):
    return min(_una(fn) for _ in range(reps))


def _una(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def main() -> None:
    n_min = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 22
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3

    cores = os.cpu_count() or 1
    workers = [w for w in WORKERS if w <= cores * 2]
    rng = np.random.default_rng(0)

    print(f"cores={cores}  reps={reps}  (se reporta el mínimo)")
    print()
    cab = f"{'n':>3} {'N':>3} {'D':>3} {'MB':>7} {'serial':>9} " + " ".join(
        f"{'w=' + str(w):>9}" for w in workers
    )
    print(cab)
    print(f"{'':>3} {'':>3} {'':>3} {'':>7} {'':>9} " + " ".join(f"{'speedup':>9}" for _ in workers))
    print("-" * len(cab))

    for n in range(n_min, n_max + 1, 2):
        sub = _sub(n, rng)
        N, D = len(sub.ncubos), len(sub.dims)
        mb = N * (1 << D) * 4 / 1e6

        t_seq = _cronometrar(lambda: preparar_oraculo(sub), reps)
        tiempos = [
            _cronometrar(lambda w=w: preparar_oraculo_mul(sub, w), reps) for w in workers
        ]

        print(
            f"{n:>3} {N:>3} {D:>3} {mb:>7.1f} {t_seq:>8.4f}s "
            + " ".join(f"{t:>8.4f}s" for t in tiempos)
        )
        print(
            f"{'':>3} {'':>3} {'':>3} {'':>7} {'1.00x':>9} "
            + " ".join(f"{t_seq / t:>8.2f}x" for t in tiempos)
        )
        del sub

    print()
    print("Ancho de banda efectivo = (D+2) pasadas × MB / tiempo.")
    print("Si el speedup se aplana muy por debajo de w, el bus está saturado y")
    print("QSW.md §6 tiene razón; si sigue subiendo, no la tenía.")


if __name__ == "__main__":
    main()
