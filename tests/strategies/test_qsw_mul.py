"""Tests de QSW-MUL: el Zeta repartido entre procesos.

La propiedad que importa es **igualdad exacta**, no `allclose`: las filas del
butterfly son independientes y el orden de sumas dentro de cada fila no cambia al
repartirlas, así que el resultado tiene que ser bit-idéntico al secuencial. Si un
día deja de serlo, es un bug, no ruido numérico.
"""

import numpy as np
import pytest

from src.iit.core.params import Params
from src.iit.strategies.python.qn.oracle import preparar_oraculo
from src.iit.strategies.python.qsw.code import QSW
from src.iit.strategies.python.qsw_mul.code import QSWMUL
from src.iit.strategies.python.qsw_mul.zeta_mul import preparar_oraculo_mul, repartir
from src.iit.strategies.python.sia import SIA
from src.iit.strategies.runner import resolver_estrategia
from src.io.manager import cargar_mpt, reducir_a_subsistema

RED = "N15A"

# Cubre el caso divisible (15/5), el que deja resto (15/4, 15/7), el degenerado
# (1 worker) y el que pide más workers que filas.
WORKERS = [1, 2, 4, 5, 7, 16, 64]


def _subsistema():
    tpm = cargar_mpt(RED)
    n = tpm.shape[1]
    return reducir_a_subsistema(
        tpm, Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)
    )


def _subsistema_sintetico(n, seed=7):
    """TPM aleatoria — el fixture N15A da pérdida 0 y no distingue nada."""
    rng = np.random.default_rng(seed)
    tpm = rng.random((1 << n, n))
    return reducir_a_subsistema(
        tpm, Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)
    )


class TestReparto:
    @pytest.mark.parametrize("n_workers", WORKERS)
    def test_cubre_todas_las_filas_sin_solaparse(self, n_workers):
        N = 15
        rangos = repartir(N, n_workers)
        assert [lo for lo, _ in rangos] == [0] + [hi for _, hi in rangos[:-1]]
        assert rangos[-1][1] == N
        assert all(hi > lo for lo, hi in rangos)

    def test_nunca_pide_mas_procesos_que_filas(self):
        # Un worker sin filas sería un proceso pagado para no hacer nada.
        assert len(repartir(15, 64)) == 15


class TestZetaParalelo:
    @pytest.mark.parametrize("n_workers", WORKERS)
    def test_bit_identico_al_secuencial(self, n_workers):
        sub = _subsistema()
        seq = preparar_oraculo(sub)
        mul = preparar_oraculo_mul(sub, n_workers)
        assert np.array_equal(seq.sumas, mul.sumas)
        assert mul.sumas.dtype == seq.sumas.dtype

    def test_el_oraculo_es_equivalente_en_todo_lo_demas(self):
        sub = _subsistema()
        seq = preparar_oraculo(sub)
        mul = preparar_oraculo_mul(sub, 4)
        assert mul.pivot_flat == seq.pivot_flat
        assert mul.pos_dim == seq.pos_dim
        assert mul.pos_idx == seq.pos_idx
        assert mul.full_mask == seq.full_mask
        assert mul.D == seq.D
        assert np.array_equal(mul.indices_order, seq.indices_order)

    def test_el_buffer_compartido_sobrevive_al_retorno(self):
        # `sumas` apunta al RawArray; si nadie retiene el buffer, esto lee basura.
        sub = _subsistema()
        mul = preparar_oraculo_mul(sub, 4)
        esperado = mul.sumas.copy()
        import gc

        gc.collect()
        assert np.array_equal(mul.sumas, esperado)


class TestSpawn:
    """El cluster es Linux, y ahí el repo usa `spawn`, no `fork`.

    Bajo `fork` el `RawArray` se hereda y el reparto funciona casi por accidente;
    bajo `spawn` tiene que viajar por `Pool(initargs=...)`, que lo transporta
    duplicando el descriptor del mmap. Es exactamente la ruta que corre en el EPYC
    y la que no se ejercita en el Mac, así que se fuerza acá.
    """

    def test_el_buffer_compartido_viaja_bajo_spawn(self, monkeypatch):
        import src.iit.base.app as appmod

        monkeypatch.setattr(appmod.aplicacion, "op_system", "linux")
        sub = _subsistema()
        seq = preparar_oraculo(sub)
        mul = preparar_oraculo_mul(sub, 4)
        assert np.array_equal(seq.sumas, mul.sumas)


class TestEquivalenciaConQSW:
    @pytest.mark.parametrize("modo", ["exacto", "estatico"])
    def test_misma_solucion_que_qsw(self, modo, monkeypatch):
        # N15A queda bajo MIN_ELEMS_PARALELO; se baja el umbral para ejercitar de
        # verdad la ruta paralela en vez de medir el fallback.
        monkeypatch.setattr(
            "src.iit.strategies.python.qsw_mul.code.MIN_ELEMS_PARALELO", 0
        )
        sub = _subsistema_sintetico(12)
        a = resolver_estrategia(sub, "qsw", {"modo": modo})
        b = resolver_estrategia(sub, "qsw_mul", {"modo": modo, "workers": "4"})
        assert a.perdida == b.perdida
        assert a.particion == b.particion
        assert a.perdida > 0  # si fuera 0 el test no distinguiría nada

    def test_workers_1_cae_a_la_ruta_secuencial(self):
        sub = _subsistema_sintetico(12)
        a = resolver_estrategia(sub, "qsw", {"modo": "exacto"})
        b = resolver_estrategia(sub, "qsw_mul", {"modo": "exacto", "workers": "1"})
        assert a.perdida == b.perdida

    def test_bajo_el_umbral_no_levanta_procesos(self, monkeypatch):
        """Debajo de MIN_ELEMS_PARALELO tiene que usar el camino de `qsw`."""
        llamado = []
        monkeypatch.setattr(
            "src.iit.strategies.python.qsw_mul.code.preparar_oraculo_mul",
            lambda *a, **k: llamado.append(1),
        )
        inst = QSWMUL(_subsistema())  # N=D=15 → 491520 < 2^20
        inst.aplicar_opciones({"workers": "8"})
        inst._preparar_oraculo(inst.sistema)
        assert llamado == []


class TestOpciones:
    def test_registrada(self):
        assert SIA.registry["qsw_mul"] is QSWMUL

    def test_no_admite_backend(self):
        # El kernel C no participa de esta ruta: aceptar `backend=c` etiquetaría el
        # CSV con algo que no corrió.
        with pytest.raises(ValueError, match="no admite la opción"):
            QSWMUL.validar_opciones({"backend": "c"})

    def test_hereda_modo_y_k_de_qsw(self):
        assert QSWMUL.opciones["modo"] == QSW.opciones["modo"]
        assert QSWMUL.opciones["k"] == QSW.opciones["k"]

    def test_workers_default_es_auto(self):
        assert QSWMUL.defaults()["workers"] == "auto"

    def test_workers_invalido_lanza(self):
        with pytest.raises(ValueError, match="inválido"):
            QSWMUL.validar_opciones({"workers": "7"})

    def test_auto_resuelve_a_los_cores(self):
        import os

        inst = QSWMUL(_subsistema())
        assert inst.n_workers() == max(1, os.cpu_count() or 1)
        inst.aplicar_opciones({"workers": "4"})
        assert inst.n_workers() == 4

    def test_preflight_ya_no_es_esqueleto(self):
        QSWMUL.preflight({"modo": "estatico"})  # no debe lanzar
