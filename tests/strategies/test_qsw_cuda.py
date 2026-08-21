"""Tests de QSW-CUDA sin GPU.

No hay NVIDIA en la máquina de desarrollo, así que el kernel no se puede ejecutar.
Lo que sí se puede validar —y es lo que más probablemente esté mal— es la
**aritmética de índices** del kernel: se replica en numpy, elemento por elemento
igual que el .cu, y se exige que reproduzca `zeta_inplace` de forma exacta.

Si `_simular_kernel` y el fuente CUDA se desincronizan, este test deja de proteger
nada: cualquier cambio a `_ZETA_DIM_SRC` tiene que reflejarse acá.
"""

import numpy as np
import pytest

from src.iit.strategies.python.qsw.code import QSW
from src.iit.strategies.python.qsw_cuda.code import QSWCUDA
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.python.zeta import zeta_inplace

def _simular_kernel(flat, N, D, pivot_flat):
    """Réplica en numpy de `_ZETA_DIM_SRC`, con su misma aritmética de índices.

        lo = ((j >> d) << (d + 1)) | (j & (2^d - 1))

    Dentro de un paso `d` cada elemento pertenece a exactamente un par y sólo uno
    de los dos lados se escribe, así que no hay carrera entre threads — y por eso
    la versión vectorizada de numpy es equivalente a la paralela de la GPU.
    """
    total = 1 << D
    pares_por_fila = total >> 1
    plano = flat.reshape(-1)

    i = np.arange(N * pares_por_fila, dtype=np.int64)
    fila = i // pares_por_fila
    j = i % pares_por_fila

    for d in range(D):
        bloque = 1 << d
        lo = ((j >> d) << (d + 1)) | (j & (bloque - 1))
        base = fila * total + lo
        if (pivot_flat >> d) & 1:
            plano[base] += plano[base + bloque]
        else:
            plano[base + bloque] += plano[base]
    return flat


class TestAritmeticaDelKernel:
    @pytest.mark.parametrize("D", [1, 2, 3, 5, 8])
    @pytest.mark.parametrize("N", [1, 4])
    def test_reproduce_zeta_inplace_en_todo_pivote(self, D, N):
        rng = np.random.default_rng(0)
        base = rng.random((N, 1 << D)).astype(np.float32)
        for pivot_flat in range(1 << D):
            esperado = zeta_inplace(base.copy(), N, D, pivot_flat)
            obtenido = _simular_kernel(base.copy(), N, D, pivot_flat)
            assert np.array_equal(esperado, obtenido), (D, N, pivot_flat)

    def test_cada_par_se_toca_una_sola_vez(self):
        """Si un índice se repitiera, en la GPU sería una carrera de escritura."""
        D, N = 6, 3
        total, pares = 1 << D, (1 << D) >> 1
        i = np.arange(N * pares, dtype=np.int64)
        fila, j = i // pares, i % pares
        for d in range(D):
            bloque = 1 << d
            lo = ((j >> d) << (d + 1)) | (j & (bloque - 1))
            base = fila * total + lo
            tocados = np.concatenate([base, base + bloque])
            assert len(np.unique(tocados)) == len(tocados)
            assert len(tocados) == N * total  # cubre el arreglo entero


class TestRegistroYOpciones:
    def test_registrada(self):
        assert SIA.registry["qsw_cuda"] is QSWCUDA

    def test_no_admite_backend(self):
        with pytest.raises(ValueError, match="no admite la opción"):
            QSWCUDA.validar_opciones({"backend": "c"})

    def test_hereda_modo_y_k(self):
        assert QSWCUDA.opciones["modo"] == QSW.opciones["modo"]
        assert QSWCUDA.opciones["k"] == QSW.opciones["k"]

    def test_preflight_falla_explicito_sin_cupy(self):
        """Sin GPU tiene que abortar antes del barrido, no fila por fila — y nunca
        caer a numpy en silencio bajo la etiqueta `qsw_cuda`."""
        try:
            import cupy  # noqa: F401
        except ImportError:
            pass
        else:
            pytest.skip("cupy instalado: el hard-fail no aplica acá")

        with pytest.raises(RuntimeError, match="requiere CUDA"):
            QSWCUDA.preflight({"modo": "estatico"})

    def test_opcion_invalida_se_valida_antes_de_mirar_la_gpu(self):
        # El orden importa: un typo debe dar el error del typo, no "falta cupy".
        with pytest.raises(ValueError, match="inválido"):
            QSWCUDA.preflight({"modo": "turbo"})
