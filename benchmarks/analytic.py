"""Benchmark completo: 6 estrategias Analytic."""

import time
import numpy as np
from itertools import combinations

from src.iit.core.ncube import NCube
from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import Analytic
from src.iit.strategies.python.analytic_parallel.code import AnalyticParallel
from src.iit.strategies.python.analytic.code_optimized import AnalyticOptimized
from src.iit.strategies.python.analytic.code_hybrid import AnalyticHybrid
from src.iit.strategies.python.analytic.code_parallel_persistent import (
    AnalyticParallelPersistent,
)
from src.iit.strategies.python.analytic.code_numba import AnalyticNumba


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

    # Secuencial
    t0 = time.perf_counter()
    seq = Analytic(sistema)
    alcance_seq, mecanismo_seq = seq.winner(sistema)
    t_seq = time.perf_counter() - t0

    # Paralelo
    t0 = time.perf_counter()
    par = AnalyticParallel(sistema, n_workers=n_workers)
    alcance_par, mecanismo_par = par.winner(sistema)
    t_par = time.perf_counter() - t0

    # Optimizado
    t0 = time.perf_counter()
    opt = AnalyticOptimized(sistema)
    alcance_opt, mecanismo_opt = opt.winner(sistema)
    t_opt = time.perf_counter() - t0

    # Híbrido
    t0 = time.perf_counter()
    hyb = AnalyticHybrid(sistema, n_workers=n_workers)
    alcance_hyb, mecanismo_hyb = hyb.winner(sistema)
    t_hyb = time.perf_counter() - t0

    # Persistent (primera llamada crea el pool)
    t0 = time.perf_counter()
    per = AnalyticParallelPersistent(sistema, n_workers=n_workers)
    alcance_per, mecanismo_per = per.winner(sistema)
    t_per = time.perf_counter() - t0

    # Persistent segunda llamada (pool ya creado)
    t0 = time.perf_counter()
    per2 = AnalyticParallelPersistent(sistema, n_workers=n_workers)
    alcance_per2, mecanismo_per2 = per2.winner(sistema)
    t_per2 = time.perf_counter() - t0

    # Numba (primera llamada compila)
    t0 = time.perf_counter()
    num = AnalyticNumba(sistema)
    alcance_num, mecanismo_num = num.winner(sistema)
    t_num = time.perf_counter() - t0

    # Numba segunda llamada (cacheada)
    t0 = time.perf_counter()
    num2 = AnalyticNumba(sistema)
    alcance_num2, mecanismo_num2 = num2.winner(sistema)
    t_num2 = time.perf_counter() - t0

    num_part = sum(len(list(combinations(range(D), k))) for k in range(1, D // 2 + 1))

    def sp(t):
        return t_seq / t if t > 0 else 0

    print(
        f"D={D:2d} N={N:6d} P={num_part:4d} | "
        f"Seq: {t_seq * 1000:7.2f} | "
        f"Par: {t_par * 1000:7.2f}({sp(t_par):4.1f}x) | "
        f"Opt: {t_opt * 1000:7.2f}({sp(t_opt):4.1f}x) | "
        f"Per: {t_per * 1000:7.2f}({sp(t_per):4.1f}x) | "
        f"Per2: {t_per2 * 1000:6.2f}({sp(t_per2):4.1f}x) | "
        f"Num: {t_num * 1000:7.2f}({sp(t_num):4.1f}x) | "
        f"Num2: {t_num2 * 1000:6.2f}({sp(t_num2):4.1f}x)"
    )

    assert (
        alcance_seq
        == alcance_par
        == alcance_opt
        == alcance_hyb
        == alcance_per
        == alcance_per2
        == alcance_num
        == alcance_num2
    )
    assert (
        mecanismo_seq
        == mecanismo_par
        == mecanismo_opt
        == mecanismo_hyb
        == mecanismo_per
        == mecanismo_per2
        == mecanismo_num
        == mecanismo_num2
    )

    return t_seq, t_par, t_opt, t_hyb, t_per, t_per2, t_num, t_num2


if __name__ == "__main__":
    print("=" * 160)
    print("Benchmark: 6 Estrategias Analytic (D=5..10)")
    print("=" * 160)
    print(
        "Seq=Secuencial  Par=ThreadPool  Opt=Optimizado  Per=Persistent  Per2=Persistent(2da)  Num=Numba  Num2=Numba(2da)"
    )
    print("=" * 160)

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

    print("=" * 160)

    # Cerrar pool persistente
    AnalyticParallelPersistent.shutdown_pool()
