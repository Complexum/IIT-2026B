"""Estrategia QSW-CUDA: `qsw` con la transformada Zeta en un kernel CUDA propio.

Mismo algoritmo que `qsw`: esta clase sobrescribe únicamente `_preparar_oraculo`.
La búsqueda —MAO, candidatas, re-scoring exacto, reconstrucción con EMD real— se
hereda sin tocar, así que el resultado es idéntico.

Qué va a la GPU, y por qué eso y no otra cosa
---------------------------------------------
El Zeta es el 100 % del costo (n=22: 0.957 s contra 2.8 ms de búsqueda, `QSW.md`
§6), y encima está limitado por ancho de banda: en CPU corre a 17-19 GB/s, o sea
saturando DRAM. Ése es exactamente el caso que una GPU con HBM resuelve, y es lo
que `QSW.md` §8 anota como pendiente.

**El error que esto evita:** `analytic_cuda` calcula `hyperfaces` —el Zeta entero,
`Θ(D·N·2^D)`— en el *host*, y manda a la GPU sólo la reducción por máscara, que ya
era barata. La GPU queda casi ociosa y el perfilador marca ~0 % de uso. Acá el
butterfly entero corre en la GPU y el host sólo arma el arreglo y lee el resultado.

Cuentas a D=20, N=20 (84 MB), para saber qué esperar::

    CPU   20 pasadas × 84 MB / 18 GB/s              ≈ 89 ms
    GPU   H2D 84 MB + 20 pasadas en HBM + D2H 84 MB ≈ 16 ms   → ~5.5x

Segunda iteración posible: dejar `sumas` residente en GPU y portar `f_cara_batch` a
cupy. Ahorra el D2H (≈ 7 ms de los 16) pero duplica el oráculo; hoy no paga.

Cómo verificar que la GPU realmente trabajó
-------------------------------------------
`self._gpu_ms` guarda el tiempo del butterfly medido con `cupy.cuda.Event`. La
columna `gpu_mem_mb` del CSV **no** sirve para esto: `pynvml` no es dependencia del
proyecto, así que `NVML_AVAILABLE` es False y la columna vale 0.0 siempre, corra o
no la GPU. Y aun con `pynvml` mide un delta de memoria, que el memory pool de cupy
deja en 0 a partir de la segunda fila. Para uso real: `nvidia-smi dmon` o `nsys`.

Sin cupy o sin dispositivo **falla explícito**, nunca degrada a CPU en silencio: el
nombre de la estrategia acaba en el CSV, y correr numpy bajo la etiqueta
`qsw_cuda` haría que la comparación mienta.

Instalación en el nodo con GPU: `uv pip install -e ".[cuda]"`.
"""

import numpy as np

from src.iit.strategies.python.qn.oracle import Oraculo
from src.iit.strategies.python.qsw.code import QSW
from src.iit.strategies.python.zeta import pivote_plano

#: Cota dura por memoria: `sumas` son N·2^D float32 en VRAM. Además de esto se
#: comprueba la memoria libre real del dispositivo, que es lo que de verdad manda.
D_MAX_CUDA = 27

_BLOCK = 256

# Un thread por par del butterfly. `bloque = 2^d` son los elementos contiguos de
# cada mitad; el destino de la suma es el lado OPUESTO al bit del pivote, porque el
# arreglo vive en coordenadas delta (la cara lógica `m` está en `pivot ^ m`) —
# misma convención que `zeta.zeta_inplace` y `clang/qsw/code.c`.
#
# Índice: el par `j` de la fila toca los elementos `lo` y `lo + 2^d`, con
# `lo = (j >> d) << (d+1) | (j & (2^d - 1))`: los bits altos de `j` eligen el grupo
# y los bajos el desplazamiento dentro de la mitad.
_ZETA_DIM_SRC = r"""
extern "C" __global__
void zeta_dim(float* __restrict__ flat,
              const long long pares_por_fila,
              const long long total_pares,
              const long long total,
              const int d,
              const int piv_bit)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total_pares) return;

    long long fila = i / pares_por_fila;
    long long j    = i % pares_por_fila;

    const long long bloque = 1LL << d;
    long long lo = ((j >> d) << (d + 1)) | (j & (bloque - 1));

    float* p = flat + fila * total + lo;
    if (piv_bit) p[0] += p[bloque];
    else         p[bloque] += p[0];
}
"""


class QSWCUDA(QSW, nombre="qsw_cuda"):
    """`qsw` con el precómputo Zeta en la GPU."""

    #: Sin `backend`: el kernel C es la otra ruta y no participa de ésta.
    opciones = {"modo": QSW.opciones["modo"], "k": QSW.opciones["k"]}

    @classmethod
    def _cupy(cls):
        """Importa cupy y exige un dispositivo. Lanza con un mensaje accionable."""
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("sin dispositivos CUDA")
        except Exception as e:
            raise RuntimeError(
                "qsw_cuda requiere CUDA + cupy. Instalá '.[cuda]' en un nodo con GPU. "
                "Para CPU usá 'qsw' o 'qsw_mul'."
            ) from e
        return cp

    @classmethod
    def preflight(cls, opciones: dict[str, str] | None = None) -> None:
        """Falla antes del barrido, no una vez por fila."""
        cls.validar_opciones(opciones)
        cls._cupy()

    def _preparar_oraculo(self, sistema):
        cp = self._cupy()

        ncubos = sistema.ncubos
        N, D = len(ncubos), len(sistema.dims)
        total = 1 << D

        if D > D_MAX_CUDA:
            raise RuntimeError(
                f"qsw_cuda: D={D} excede D_MAX_CUDA={D_MAX_CUDA} (memoria GPU)."
            )
        libre, _ = cp.cuda.Device().mem_info
        necesita = N * total * 4
        if necesita > libre:
            raise RuntimeError(
                f"qsw_cuda: hacen falta {necesita / 1e9:.2f} GB en VRAM para "
                f"sumas (N={N}, D={D}) y hay {libre / 1e9:.2f} GB libres."
            )

        # Host: (N, 2^D) float32 directo, sin el stack float64 intermedio que hace
        # `zeta_caras` (pico de 12·N·2^D bytes).
        flat = np.empty((N, total), dtype=np.float32)
        for i, c in enumerate(ncubos):
            flat[i] = c.data

        pivot_flat = pivote_plano(sistema)

        g = cp.asarray(flat)
        # δ = H − p con el pivote en 0, igual que la ruta numpy.
        g -= g[:, pivot_flat][:, None]

        kernel = cp.RawKernel(_ZETA_DIM_SRC, "zeta_dim")
        pares_por_fila = total >> 1
        total_pares = N * pares_por_fila
        grid = ((total_pares + _BLOCK - 1) // _BLOCK,)

        ini, fin = cp.cuda.Event(), cp.cuda.Event()
        ini.record()
        for d in range(D):
            kernel(
                grid,
                (_BLOCK,),
                (
                    g,
                    np.int64(pares_por_fila),
                    np.int64(total_pares),
                    np.int64(total),
                    np.int32(d),
                    np.int32((pivot_flat >> d) & 1),
                ),
            )
        fin.record()
        fin.synchronize()
        #: Prueba de que el kernel corrió — ver el docstring del módulo.
        self._gpu_ms = float(cp.cuda.get_elapsed_time(ini, fin))

        sumas = cp.asnumpy(g)
        del g

        indices_order = np.fromiter((c.indice for c in ncubos), dtype=np.int64)
        return Oraculo(
            sumas=sumas,
            pos_dim={d: i for i, d in enumerate(sistema.dims)},
            indices_order=indices_order,
            full_mask=total - 1,
            D=D,
            pos_idx={int(idx): i for i, idx in enumerate(indices_order)},
            pivot_flat=pivot_flat,
        )
