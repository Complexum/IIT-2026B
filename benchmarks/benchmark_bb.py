"""Benchmark: Analytic vs Branch-and-Bound."""

import time
import numpy as np

from src.iit.core.ncube import NCube
from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import Analytic
from src.iit.strategies.python.analytic_bb.code import AnalyticBB


def generar_sistema_aleatorio(D: int, N: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    ncubos = []
    for i in range(N):
        data = rng.random(2**D).astype(np.float32)
        ncubos.append(NCube(i, tuple(range(D)), data))
    estado = tuple(0 for _ in range(D))
    return System(estado, tuple(ncubos))


def benchmark(D: int, N: int):
    sistema = generar_sistema_aleatorio(D, N)

    # Analytic original
    t0 = time.perf_counter()
    seq = Analytic(sistema)
    alcance_seq, mecanismo_seq = seq.winner(sistema)
    t_seq = time.perf_counter() - t0

    # Branch-and-Bound
    t0 = time.perf_counter()
    bb = AnalyticBB(sistema)
    alcance_bb, mecanismo_bb = bb.winner(sistema)
    t_bb = time.perf_counter() - t0

    match = alcance_seq == alcance_bb and mecanismo_seq == mecanismo_bb

    print(
        f"D={D:2d} N={N:4d} | "
        f"Analytic: {t_seq * 1000:8.2f}ms | "
        f"BB: {t_bb * 1000:8.2f}ms | "
        f"Match: {'✅' if match else '❌'}"
    )

    return t_seq, t_bb, match


if __name__ == "__main__":
    print("=" * 80)
    print("Benchmark: Analytic vs Branch-and-Bound")
    print("=" * 80)

    casos = [
        (2, 5),
        (3, 5),
        (4, 5),
        (5, 5),
        (6, 5),
        (7, 5),
        (8, 5),
    ]

    for D, N in casos:
        try:
            benchmark(D, N)
        except Exception as e:
            print(f"D={D} N={N} ERROR: {e}")

    print("=" * 80)
