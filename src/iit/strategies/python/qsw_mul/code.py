"""Estrategia QSW-MUL: `qsw` con la transformada Zeta repartida en multiprocessing.

Mismo algoritmo que `qsw` —mismo MAO, mismas candidatas, mismo re-scoring exacto y
la misma reconstrucción con `bipartir + distribucion_marginal + emd_efecto`— y por
eso el resultado es **bit-idéntico**: esta clase sobrescribe únicamente
`_preparar_oraculo`, el precómputo. La búsqueda se hereda sin tocarla.

Qué se paraleliza y por qué sólo eso
------------------------------------
El Zeta es el 100 % del costo: a n=22 son 0.957 s contra 2.8 ms de búsqueda
(`QSW.md` §6). Repartir el MAO sería repartir 2.8 ms pagando IPC. Las N filas del
butterfly son independientes (`QSW.md` §8), así que van a `multiprocessing` sobre
memoria compartida — ver `zeta_mul.py` para el porqué de `RawArray`.

La hipótesis que esto mide
--------------------------
`QSW.md` §6 cierra diciendo que, por estar limitado por ancho de banda, más hilos
en la misma máquina compran poco. Es una hipótesis, no un hecho: 17–19 GB/s es ~7 %
de lo que da un M4 Pro (~273 GB/s), o sea que **un core está lejos de saturar la
DRAM**. `benchmarks/escalado_mul.py` produce la curva que lo confirma o lo refuta.

Techos conocidos, dichos por adelantado:

- El reparto es por filas, así que está acotado por N. Con N=20 y 16 workers quedan
  4 rebanadas de 2 filas y 12 de 1 → ~10× efectivo, no 16×.
- El llenado y la resta del pivote quedan seriales: 2 pasadas contra D → Amdahl
  ~11× a D=20.
- `ResourceMonitor` mide con `RUSAGE_SELF`, que no cuenta procesos hijos, así que
  `tiempo_cpu_s` y `cpu_user_s` van a sub-reportar para esta estrategia.
  `tiempo_wall_s` —la métrica de speedup— sí es correcta.

Opciones: las de `qsw` menos `backend`, más `workers`.
"""

import os

from src.iit.strategies.python.qsw.code import QSW
from src.iit.strategies.python.qsw_mul.zeta_mul import preparar_oraculo_mul

#: Debajo de este tamaño el Zeta se hace serial. Fijado con `benchmarks/escalado_mul.py`
#: sobre M4 Pro (12 cores), no a ojo — el reparto sólo empieza a pagar cerca de los
#: 80 MB:
#:
#:     n=16  (4.2 MB)   serial 0.0062s   w=8 0.0155s   → 0.40x  (pierde)
#:     n=18 (18.9 MB)   serial 0.0289s   w=8 0.0258s   → 1.12x  (empata)
#:     n=20 (83.9 MB)   serial 0.1423s   w=8 0.0735s   → 1.94x  (gana)
#:
#: `Pool` con `fork` cuesta 2.5-11 ms según los workers, así que el umbral no lo
#: manda el overhead del pool sino que las filas chicas caben en L2 y la versión
#: serial ya las recorre sin tocar DRAM.
MIN_ELEMS_PARALELO = 1 << 23


class QSWMUL(QSW, nombre="qsw_mul"):
    """`qsw` con el precómputo Zeta repartido entre procesos."""

    #: `backend` se omite a propósito: el kernel C no participa de esta ruta, y
    #: `--opcion backend=c` debe fallar en vez de correr algo distinto de lo que
    #: dice la etiqueta del CSV (mismo criterio que `qsw.backend.resolver_backend`).
    opciones = {
        "modo": QSW.opciones["modo"],
        "k": QSW.opciones["k"],
        "workers": ("auto", "1", "2", "4", "8", "12", "16", "32"),
    }

    workers: str = "auto"

    def n_workers(self) -> int:
        """`auto` → todos los cores. Nunca menos de 1."""
        if self.workers == "auto":
            return max(1, os.cpu_count() or 1)
        return max(1, int(self.workers))

    def _preparar_oraculo(self, sistema):
        n_workers = self.n_workers()
        N, D = len(sistema.ncubos), len(sistema.dims)

        if n_workers <= 1 or N * (1 << D) < MIN_ELEMS_PARALELO:
            return super()._preparar_oraculo(sistema)

        return preparar_oraculo_mul(sistema, n_workers)
