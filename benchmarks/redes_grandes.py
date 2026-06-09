"""Benchmark para redes grandes: D=15 y D=20."""

import time
import numpy as np
from itertools import combinations

from src.iit.core.ncube import NCube
from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import Analytic
from src.iit.strategies.python.analytic_concurrent.code import AnalyticParallel


def generar_sistema_aleatorio(D: int, N: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    ncubos = []
    for i in range(N):
        data = rng.random(2**D).astype(np.float32)
        ncubos.append(NCube(i, tuple(range(D)), data))
    estado = tuple(0 for _ in range(D))
    return System(estado, tuple(ncubos))


def benchmark(D: int, N: int, n_workers: int = 4):
    print(f"Generando sistema D={D}, N={N}...")
    sistema = generar_sistema_aleatorio(D, N)
    num_part = sum(len(list(combinations(range(D), k))) for k in range(1, D // 2 + 1))
    print(f"Particiones a evaluar: {num_part}")

    # Secuencial
    print("Ejecutando secuencial...")
    t0 = time.perf_counter()
    seq = Analytic(sistema)
    alcance_seq, mecanismo_seq = seq.winner(sistema)
    t_seq = time.perf_counter() - t0
    print(f"  Secuencial: {t_seq:.2f}s")

    # Paralelo
    print("Ejecutando paralelo...")
    t0 = time.perf_counter()
    par = AnalyticParallel(sistema, n_workers=n_workers)
    alcance_par, mecanismo_par = par.winner(sistema)
    t_par = time.perf_counter() - t0
    print(f"  Paralelo: {t_par:.2f}s")

    speedup = t_seq / t_par if t_par > 0 else 0

    print(
        f"\nD={D:2d} N={N:6d} P={num_part:5d} | "
        f"Seq: {t_seq:8.2f}s | Par: {t_par:8.2f}s | Speedup: {speedup:.2f}x"
    )

    assert alcance_seq == alcance_par, f"Discrepancia: {alcance_seq} vs {alcance_par}"
    assert mecanismo_seq == mecanismo_par, (
        f"Discrepancia: {mecanismo_seq} vs {mecanismo_par}"
    )

    return t_seq, t_par, speedup


if __name__ == "__main__":
    print("=" * 80)
    print("Benchmark Redes Grandes: D=15 y D=20")
    print("=" * 80)

    casos = [
        (15, 10),
        (15, 50),
        (15, 100),
        (20, 10),
        (20, 50),
    ]

    for D, N in casos:
        try:
            benchmark(D, N, n_workers=4)
            print()
        except Exception as e:
            print(f"D={D} N={N} ERROR: {e}\n")

    print("=" * 80)
