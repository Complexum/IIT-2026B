"""Benchmark extendido: D=5..10 para el documento técnico."""

import time
import numpy as np
from itertools import combinations

from src.iit.core.ncube import NCube
from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import Analytic
from src.iit.strategies.python.analytic.code_parallel import AnalyticParallel
from src.iit.strategies.python.analytic.code_optimized import AnalyticOptimized
from src.iit.strategies.python.analytic.code_hybrid import AnalyticHybrid


def generar_sistema_aleatorio(D: int, N: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    ncubos = []
    for i in range(N):
        data = rng.random(2**D).astype(np.float32)
        ncubos.append(NCube(i, tuple(range(D)), data))
    estado = tuple(0 for _ in range(D))
    return System(estado, tuple(ncubos))


def benchmark(D: int, N: int, n_workers: int = 4):
    sistema = generar_sistema_aleatorio(D, N)

    t0 = time.perf_counter()
    seq = Analytic(sistema)
    alcance_seq, mecanismo_seq = seq.winner(sistema)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    par = AnalyticParallel(sistema, n_workers=n_workers)
    alcance_par, mecanismo_par = par.winner(sistema)
    t_par = time.perf_counter() - t0

    t0 = time.perf_counter()
    opt = AnalyticOptimized(sistema)
    alcance_opt, mecanismo_opt = opt.winner(sistema)
    t_opt = time.perf_counter() - t0

    t0 = time.perf_counter()
    hyb = AnalyticHybrid(sistema, n_workers=n_workers)
    alcance_hyb, mecanismo_hyb = hyb.winner(sistema)
    t_hyb = time.perf_counter() - t0

    sp_par = t_seq / t_par if t_par > 0 else 0
    sp_opt = t_seq / t_opt if t_opt > 0 else 0
    sp_hyb = t_seq / t_hyb if t_hyb > 0 else 0

    num_part = sum(len(list(combinations(range(D), k))) for k in range(1, D // 2 + 1))

    print(
        f"D={D:2d} N={N:6d} P={num_part:4d} | "
        f"Seq: {t_seq * 1000:8.2f}ms | "
        f"Par: {t_par * 1000:8.2f}ms ({sp_par:4.1f}x) | "
        f"Opt: {t_opt * 1000:8.2f}ms ({sp_opt:4.1f}x) | "
        f"Hyb: {t_hyb * 1000:8.2f}ms ({sp_hyb:4.1f}x)"
    )

    assert alcance_seq == alcance_par
    assert mecanismo_seq == mecanismo_par
    assert alcance_seq == alcance_opt
    assert mecanismo_seq == mecanismo_opt
    assert alcance_seq == alcance_hyb
    assert mecanismo_seq == mecanismo_hyb

    return t_seq, t_par, t_opt, t_hyb


if __name__ == "__main__":
    print("=" * 130)
    print("Benchmark Extendido: Analytic Estrategias (D=5..10)")
    print("=" * 130)

    casos = [
        (5, 100),
        (5, 500),
        (5, 2000),
        (6, 100),
        (6, 500),
        (6, 2000),
        (7, 100),
        (7, 500),
        (7, 2000),
        (8, 100),
        (8, 500),
        (8, 2000),
        (9, 100),
        (9, 500),
        (10, 100),
        (10, 500),
    ]

    for D, N in casos:
        try:
            benchmark(D, N, n_workers=4)
        except Exception as e:
            print(f"D={D} N={N} ERROR: {e}")

    print("=" * 130)
