"""Estrategia AnalyticCUDA: MIP óptimo signed-δ con kernel CUDA propio (cupy RawKernel).

Misma matemática óptima que `analytical` (normalización SIGNADA δ = H − p, pivote→0,
valor absoluto aplicado UNA vez tras promediar). La fase pesada —costo por máscara y
reducción Σ_i— se evalúa en un **kernel CUDA propio**, un bloque por máscara, reducción
en memoria compartida sobre los hipercubos. Ver `demo.md` (Sección 8) para el modelo.

Pipeline
--------
1. Zeta transform (`hyperfaces`) sobre δ → `sumas[i, m]` (host, O(D·N·2^D)).
2. `sumas` → GPU; kernel `mask_cost` calcula
       f(m) = Σ_i min(|sumas[i,m]/2^popc(m)|, |sumas[i,m^full]/2^(D−popc(m))|)
   para m ∈ [1, 2^(D-1)]  (m = blockIdx.x + 1).
3. argmin en GPU; derivación (alcance, mecanismo) y EMD real en host.

Hard-fail: sin `cupy`/GPU, `resolver` lanza `RuntimeError` (la estrategia sigue
registrada y visible en CLI). Instalación cluster: `uv pip install -e ".[cuda]"`.
Cota memoria: D ≤ D_MAX_CUDA (sumas (N, 2^D) float32 en VRAM).
"""

import time

import numpy as np

from src.iit.base.consts import FLOAT_ZERO, INFTY_POS
from src.iit.base.funcs import emd_efecto
from src.iit.core.solution import Solution
from src.iit.core.system import System
from src.iit.strategies.python.analytic.code import hyperfaces
from src.iit.strategies.python.fmt import fmt_parts
from src.iit.strategies.python.sia import SIA

D_MAX_CUDA = 27  # sumas (N, 2^D) float32; cota de memoria GPU
_BLOCK = 256  # threads por bloque (potencia de 2 para la reducción en árbol)

# Kernel: un bloque por máscara m = blockIdx.x + 1; threads reducen sobre i.
_MASK_COST_SRC = r"""
extern "C" __global__
void mask_cost(const float* __restrict__ sumas,
               const long long N, const int D,
               const unsigned int full, const unsigned long long S,
               const int M, float* __restrict__ out)
{
    int mb = blockIdx.x;
    if (mb >= M) return;

    unsigned int m  = (unsigned int)(mb + 1);
    unsigned int mc = full ^ m;
    int a  = __popc(m);
    float invA = exp2f(-(float)a);
    float invB = exp2f(-(float)(D - a));

    float acc = 0.0f;
    for (long long i = threadIdx.x; i < N; i += blockDim.x) {
        size_t base = (size_t)i * (size_t)S;
        float vA = sumas[base + m]  * invA;
        float vB = sumas[base + mc] * invB;
        float fa = fabsf(vA);
        float fb = fabsf(vB);
        acc += (fa < fb ? fa : fb);
    }

    __shared__ float sdata[256];
    int t = threadIdx.x;
    sdata[t] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (t < s) sdata[t] += sdata[t + s];
        __syncthreads();
    }
    if (t == 0) out[mb] = sdata[0];
}
"""


class AnalyticCUDA(SIA, nombre="analytic_cuda"):
    """MIP óptimo signed-δ con evaluación de máscaras en kernel CUDA propio (cupy)."""

    def winner(
        self, sistema: System, cp, kernel
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        D = len(sistema.dims)
        N = len(sistema.ncubos)

        data_nd = np.stack([c.ndata for c in sistema.ncubos])
        pivot_idx = tuple(int(sistema.estado_inicial[dim]) for dim in sistema.dims)
        pivot_vals = data_nd[(slice(None),) + pivot_idx]  # (N,)

        # Normalización SIGNADA: δ = H − p, el pivote queda en 0.
        delta_nd = data_nd - pivot_vals.reshape((N,) + (1,) * D)

        # C_conc = |mean_full(δ)| = |mean_full(H) − p| para colapso
        all_mean = delta_nd.reshape(N, -1).mean(axis=1)
        conc_costs = np.abs(all_mean)
        conc_idx = int(np.argmin(conc_costs))
        C_conc = float(conc_costs[conc_idx])

        if D <= 1:
            return (sistema.ncubos[conc_idx].indice,), ()

        full_mask = (1 << D) - 1
        S = 1 << D
        M = full_mask >> 1  # máscaras 1..2^(D-1); m = blockIdx.x + 1

        # ── 1. Zeta transform sobre δ (host) → sumas (N, 2^D) ─────────────────
        sumas = hyperfaces(N, D, delta_nd, pivot_idx)  # float32

        # ── 2. Kernel CUDA: f(m) por máscara ──────────────────────────────────
        sumas_g = cp.ascontiguousarray(cp.asarray(sumas, dtype=cp.float32))
        f_all_g = cp.empty(M, dtype=cp.float32)
        kernel(
            (M,),
            (_BLOCK,),
            (
                sumas_g,
                np.int64(N),
                np.int32(D),
                np.uint32(full_mask),
                np.uint64(S),
                np.int32(M),
                f_all_g,
            ),
        )

        best_idx = int(cp.argmin(f_all_g).item())
        best_val = float(f_all_g[best_idx].item())
        best_mask_a = best_idx + 1
        best_mask_b = full_mask ^ best_mask_a

        # ── 3. Colapso vs distribución + derivación en host ───────────────────
        if C_conc <= best_val:
            return (sistema.ncubos[conc_idx].indice,), ()

        def __derive(mask_a: int) -> tuple[tuple, tuple]:
            mask_b = full_mask ^ mask_a
            sz_a = bin(mask_a).count("1")
            sz_b = D - sz_a
            va = sumas[:, mask_a] / (1 << sz_a)
            vb = sumas[:, mask_b] / (1 << sz_b)
            # Hipercubo va a A si |mean_B(δ)| <= |mean_A(δ)| (pivote ya en 0)
            in_a = np.abs(vb) <= np.abs(va)
            alc = tuple(c.indice for i, c in enumerate(sistema.ncubos) if in_a[i])
            mec = tuple(sistema.dims[d] for d in range(D) if (mask_a >> d) & 1)
            return alc, mec

        alc_a, mec_a = __derive(best_mask_a)
        alc_b, mec_b = __derive(best_mask_b)

        dm_orig = sistema.distribucion_marginal()
        emd_a = emd_efecto(
            sistema.bipartir(alc_a, mec_a).distribucion_marginal(), dm_orig
        )
        emd_b = emd_efecto(
            sistema.bipartir(alc_b, mec_b).distribucion_marginal(), dm_orig
        )

        return (alc_a, mec_a) if emd_a <= emd_b else (alc_b, mec_b)

    def resolver(self) -> Solution:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("sin dispositivos CUDA")
        except Exception as e:
            raise RuntimeError(
                "analytic_cuda requiere CUDA + cupy. Instala '.[cuda]' en un nodo con GPU."
            ) from e

        dm_original = self.distribucion
        t0 = time.perf_counter()

        if not self.sistema.indices or not self.sistema.dims:
            return Solution(
                estrategia=self.nombre.capitalize(),
                perdida=FLOAT_ZERO,
                distribucion_subsistema=dm_original,
                distribucion_particion=dm_original,
                particion=fmt_parts(((), ()), ((), ())).strip(),
                tiempo_total=time.perf_counter() - t0,
                quiere_hablar=False,
            )

        D = len(self.sistema.dims)
        if D > D_MAX_CUDA:
            raise RuntimeError(
                f"analytic_cuda: D={D} excede D_MAX_CUDA={D_MAX_CUDA} (memoria GPU)."
            )

        kernel = cp.RawKernel(_MASK_COST_SRC, "mask_cost")
        alcance, mecanismo = self.winner(self.sistema, cp, kernel)

        particion_sistema = self.sistema.bipartir(alcance, mecanismo)
        dm = particion_sistema.distribucion_marginal()
        perdida = emd_efecto(dm, dm_original)

        tiempo = time.perf_counter() - t0
        texto = fmt_parts(
            (alcance, mecanismo), (self.sistema.indices, self.sistema.dims)
        )

        return Solution(
            estrategia=self.nombre.capitalize(),
            perdida=float(perdida) if perdida != INFTY_POS else FLOAT_ZERO,
            distribucion_subsistema=dm_original,
            distribucion_particion=dm,
            particion=texto.strip(),
            tiempo_total=tiempo,
            quiere_hablar=False,
        )
