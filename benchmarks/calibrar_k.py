"""Calibración del oráculo por muestreo de QSW: exactitud y speedup vs K.

Sirve para fijar `K_BASE` / `K_POR_NODO` (en `qsw/muestreo.py`) con datos y no a
ojo, y para documentar dónde el modo estocástico deja de ser confiable.

    uv run python benchmarks/calibrar_k.py [n_min] [n_max] [repeticiones]

Cada celda reporta la tasa de coincidencia con `analytic` (exacto) sobre TPMs
sintéticas y el speedup del tiempo total.
"""

import sys
import time

import numpy as np

from src.iit.core.params import Params
from src.iit.strategies.python.analytic.code import Analytic
from src.iit.strategies.python.qsw.code import QSW
from src.iit.strategies.python.qsw.muestreo import k_efectivo
from src.io.manager import reducir_a_subsistema

KS = ["256", "1024", "2048", "8192", "auto"]


class _QSWk(QSW, nombre="qsw_calib"):
    modo = "estocastico"
    backend = "python"


def _sub(n, rng):
    tpm = rng.random((1 << n, n))
    return reducir_a_subsistema(
        tpm, Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)
    )


def main() -> None:
    n_min = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    n_max = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    print(f"Calibración del muestreo — n={n_min}..{n_max}, {reps} TPMs por punto")
    print(f"{'n':>3} {'V':>4} {'analytic':>10} " + " ".join(f"{('K='+k):>16}" for k in KS))
    print("-" * (20 + 17 * len(KS)))

    for n in range(n_min, n_max + 1, 2):
        # Un subsistema a la vez: a n=24 cada TPM son ~3 GB y tenerlos todos
        # vivos hace thrashing y falsea los tiempos.
        t_ref = 0.0
        ok = dict.fromkeys(KS, 0)
        t_acc = dict.fromkeys(KS, 0.0)
        dudosas = dict.fromkeys(KS, 0)
        V = kef = 0
        for r in range(reps):
            sub = _sub(n, np.random.default_rng(1000 + n + 7919 * r))
            V = len(sub.dims) + len(sub.indices)
            kef = k_efectivo(len(sub.indices))
            t0 = time.perf_counter()
            ref = Analytic(sub).resolver().perdida
            t_ref += time.perf_counter() - t0
            for k in KS:
                _QSWk.k = k
                t0 = time.perf_counter()
                inst = _QSWk(sub)
                sol = inst.resolver()
                t_acc[k] += time.perf_counter() - t0
                if abs(sol.perdida - ref) <= 1e-6:
                    ok[k] += 1
                if not getattr(inst, "_confianza", {}).get("confiable", True):
                    dudosas[k] += 1
            del sub
        t_ref /= reps
        celdas = [
            f"{ok[k]}/{reps} {t_ref / (t_acc[k] / reps):>6.1f}x" + ("!" if dudosas[k] else " ")
            for k in KS
        ]
        print(f"{n:>3} {V:>4} {t_ref:>9.3f}s " + " ".join(f"{c:>16}" for c in celdas)
              + f"   (auto→K={kef})")

    print("\n  formato: aciertos/total  speedup      '!' = el propio modo marcó")
    print("  alguna corrida como dudosa (margen < 2σ del ruido de muestreo)")


if __name__ == "__main__":
    main()
