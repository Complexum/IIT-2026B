"""Evidencia numérica de que el objetivo EMD de QSW no es submodular.

Acompaña a la demostración formal de ``demo/``. Los dos primeros bloques son el
puente entre el Lean y el código que realmente corre: evalúan los contraejemplos
con el ``f_cara`` de ``src/iit/strategies/python/qsw/oracle.py``, no con una
reimplementación. El tercero mide cuán frecuente es la violación sobre una red
real, y el cuarto exhibe la causa estructural (interacciones de orden ≥ 2).

Uso::

    python .docs/proofs/submodularidad/numerico.py [RED]   # RED por defecto: N15A
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

# El script vive tres niveles bajo la raíz; se añade para poder invocarlo por ruta.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.iit.core.params import Params
from src.iit.strategies.python.qsw.oracle import Oraculo, f_cara, preparar_oraculo
from src.io.manager import cargar_mpt, reducir_a_subsistema

MUESTRAS = 20_000
SEMILLA = 3
TOL = 1e-5


# ── Oráculo sintético: el mismo dataclass que usa la estrategia ────────────────
def oraculo_de_caras(delta: list[float]) -> Oraculo:
    """Construye un `Oraculo` con D=2, N=1 a partir de δ sobre las 4 caras.

    δ está indexado por máscara de dims: (∅, {a}, {b}, {a,b}). `sumas` es su
    transformada Zeta (suma sobre subconjuntos), que es lo que `preparar_oraculo`
    produce a partir de una TPM. `pivot_flat=0` porque δ ya está en coordenadas
    delta (el pivote es la máscara 0).
    """
    d = np.asarray(delta, dtype=np.float32)
    z = d.copy()
    for b in range(2):  # butterfly de la Zeta, igual que `zeta_caras`
        for m in range(4):
            if m >> b & 1:
                z[m] += z[m ^ (1 << b)]
    return Oraculo(
        sumas=z.reshape(1, 4),
        pos_dim={0: 0, 1: 1},
        indices_order=np.array([0], dtype=np.int64),
        full_mask=3,
        D=2,
        pos_idx={0: 0},
        pivot_flat=0,
    )


#: Los 8 subconjuntos de V = {a, b, z}, en el orden de la tabla del README.
SUBCONJUNTOS = [
    ("{}", (), ()),
    ("{a}", (), (0,)),
    ("{b}", (), (1,)),
    ("{a,b}", (), (0, 1)),
    ("{z}", (0,), ()),
    ("{a,z}", (0,), (0,)),
    ("{b,z}", (0,), (1,)),
    ("{a,b,z}", (0,), (0, 1)),
]


def tabla(orc: Oraculo) -> dict[str, float]:
    return {n: f_cara(orc, alc, mec) for n, alc, mec in SUBCONJUNTOS}


def contraejemplo(titulo: str, delta: list[float], S: str, T: str, union: str) -> None:
    """Imprime la tabla de f y comprueba f(S∪T) + f(S∩T) > f(S) + f(T)."""
    t = tabla(oraculo_de_caras(delta))
    print(f"\n── {titulo}")
    print(f"   δ sobre las caras (∅, a, b, ab) = {tuple(delta)}")
    print("   " + "  ".join(f"f({k})={v:g}" for k, v in t.items()))
    izq, der = t[union] + t["{}"], t[S] + t[T]
    print(f"   f({union}) + f({{}}) = {izq:g}   >   {der:g} = f({S}) + f({T})")
    assert izq > der + TOL, "el contraejemplo dejó de violar submodularidad"
    print(f"   ⇒ viola submodularidad por {izq - der:g}")


# ── Tasa de violación sobre una red real ──────────────────────────────────────
def tasa_de_violacion(red: str) -> None:
    tpm = cargar_mpt(red)
    n = tpm.shape[1]
    sis = reducir_a_subsistema(tpm, Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n))
    orc = preparar_oraculo(sis)
    dims, idxs = sis.dims, sis.indices
    D, N = len(dims), len(idxs)
    V = D + N

    memo: dict[int, float] = {}

    def f(S: int) -> float:
        if S not in memo:
            mec = tuple(dims[j] for j in range(D) if (S >> j) & 1)
            alc = tuple(idxs[i] for i in range(N) if (S >> (D + i)) & 1)
            memo[S] = f_cara(orc, alc, mec)
        return memo[S]

    rng = np.random.default_rng(SEMILLA)
    viol = tot = 0
    peor = 0.0
    por_tipo = {"dim-dim": 0, "idx-idx": 0, "cruzado": 0}

    for _ in range(MUESTRAS):
        S = int(rng.integers(0, 1 << V))
        libres = [u for u in range(V) if not (S >> u) & 1]
        if len(libres) < 2:
            continue
        u, v = (int(x) for x in rng.choice(libres, 2, replace=False))
        tot += 1
        # Segunda diferencia. Es la forma ∪/∩ con S∩T = S, S∪T = S∪{u,v}.
        brecha = f(S | 1 << u | 1 << v) + f(S) - f(S | 1 << u) - f(S | 1 << v)
        if brecha > TOL:
            viol += 1
            peor = max(peor, brecha)
            tipo = "cruzado" if (u < D) != (v < D) else ("dim-dim" if u < D else "idx-idx")
            por_tipo[tipo] += 1

    print(f"\n── Tasa de violación sobre {red} (D={D}, N={N}, V={V}, {tot} muestras)")
    print(f"   violaciones: {viol}/{tot} = {100 * viol / tot:.1f} %   peor brecha: {peor:.6f}")
    print("   por tipo de par: " + "  ".join(f"{k} {n}" for k, n in por_tipo.items()))
    assert por_tipo["idx-idx"] == 0, "f debería ser modular en el alcance"
    print("   ⇒ idx-idx = 0 no es suerte: f es modular en el alcance (ver Modular.lean)")


# ── Causa estructural: interacciones de Möbius de orden ≥ 2 ───────────────────
def caracterizacion_de_mobius(D: int = 4, semillas: int = 400) -> None:
    """La media de cara es submodular ⟺ las interacciones de orden ≥ 2 son ≤ 0.

    Con δ(x) = Σ_{T⊆x} c_T, la media de cara vale δ̄(A) = Σ_{T⊆A} c_T·2^(−|T|), así
    que la segunda diferencia en (A,u,v) es Σ_{R⊆A} c_{R∪{u,v}}·2^(−|R|−2). Forzar
    c_T ≤ 0 para |T| ≥ 2 la vuelve no positiva; el valor absoluto la rompe igual.
    """
    rng = np.random.default_rng(11)

    def media_de_cara(c: np.ndarray):
        delta = np.array([sum(c[T] for T in range(1 << D) if T & x == T) for x in range(1 << D)])
        delta -= delta[0]
        z = delta.copy()
        for b in range(D):
            for m in range(1 << D):
                if m >> b & 1:
                    z[m] += z[m ^ (1 << b)]
        return lambda A: z[A] / (1 << bin(A).count("1"))

    def peor_brecha(orden2_negativo: bool, con_abs: bool) -> float:
        peor = -np.inf
        for _ in range(semillas):
            c = rng.uniform(-1, 1, 1 << D)
            c[0] = 0.0
            if orden2_negativo:
                for T in range(1 << D):
                    if bin(T).count("1") >= 2:
                        c[T] = -abs(c[T])
            m = media_de_cara(c)
            h = (lambda A: abs(m(A))) if con_abs else m
            for A in range(1 << D):
                libres = [u for u in range(D) if not (A >> u) & 1]
                for u, v in itertools.combinations(libres, 2):
                    peor = max(peor, h(A | 1 << u | 1 << v) + h(A) - h(A | 1 << u) - h(A | 1 << v))
        return peor

    print(f"\n── Caracterización de Möbius (capa de dims sola, D={D}, {semillas} δ aleatorios)")
    for etiqueta, o2n, abs_ in [
        ("c_T ≤ 0 para |T| ≥ 2, sin |·|", True, False),
        ("c_T libre,              sin |·|", False, False),
        ("c_T ≤ 0 para |T| ≥ 2, con |·|", True, True),
    ]:
        print(f"   {etiqueta}:  peor brecha = {peor_brecha(o2n, abs_):+.3e}")
    print("   ⇒ la submodularidad de la media de cara equivale a no tener interacciones")
    print("     de orden ≥ 2 positivas, y el valor absoluto la rompe incluso si las hay.")


def main() -> None:
    red = sys.argv[1] if len(sys.argv) > 1 else "N15A"
    print("Evidencia numérica: el objetivo EMD de QSW no es submodular")
    print("Los valores de f salen de `f_cara` (oracle.py), no de una reimplementación.")

    contraejemplo(
        "Contraejemplo 1 — par dim-dim (el nodo XOR)",
        [0.0, 0.0, 0.0, 1.0],
        S="{a}", T="{b}", union="{a,b}",
    )
    contraejemplo(
        "Contraejemplo 2 — par cruzado dim-índice",
        [0.0, 0.0, 0.5, -0.5],
        S="{a}", T="{z}", union="{a,z}",
    )
    tasa_de_violacion(red)
    caracterizacion_de_mobius()


if __name__ == "__main__":
    main()
