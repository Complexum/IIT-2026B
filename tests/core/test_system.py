"""Tests for System operations: condicionar, substraer, bipartir."""

from numpy._typing._array_like import NDArray


from typing import Any


import numpy as np

from src.io.manager import cargar_mpt, crear_sistema


class TestSystemOperations:
    """Test System operations match Rust behavior."""

    def test_system_creation(self):
        tpm = cargar_mpt("N3A")
        estado = (1, 0, 0)
        sistema = crear_sistema(tpm, estado)

        assert sistema.estado_inicial == estado
        assert len(sistema.ncubos) == 3
        assert sistema.indices == (0, 1, 2)

    def test_system_condicionar(self):
        tpm = cargar_mpt("N3A")
        estado = (1, 0, 0)
        sistema = crear_sistema(tpm, estado)

        condicionado = sistema.condicionar(indices=(2,))
        assert len(condicionado.ncubos) == 2
        assert 2 not in condicionado.indices

    def test_system_substraer(self):
        tpm = cargar_mpt("N3A")
        estado = (1, 0, 0)
        sistema = crear_sistema(tpm, estado)

        substraido = sistema.substraer(alcance=(0,), mecanismo=(2,))

        assert len(substraido.ncubos) == 2
        assert 0 not in substraido.indices
        assert all(2 not in c.dims for c in substraido.ncubos)

    def test_system_distribucion_marginal(self):
        tpm = cargar_mpt("N3A")
        estado = (1, 0, 0)
        sistema = crear_sistema(tpm, estado)

        dist = sistema.distribucion_marginal()
        assert isinstance(dist, np.ndarray)
        assert len(dist) == len(sistema.ncubos)
        assert np.all(dist >= 0)
        assert np.all(dist <= 1)

    def test_system_bipartir(self):
        # Generada con semilla
        tpm = cargar_mpt("N3A")
        estado = (1, 0, 0)
        sistema = crear_sistema(tpm, estado)

        bipartido = sistema.bipartir(
            alcance=(0,),
            mecanismo=(2,),
        )

        print(bipartido)

        assert len(bipartido.ncubos) == len(sistema.ncubos)

        assert np.array_equal(
            bipartido.distribucion_marginal(),
            np.array([0.5, 0, 0.5]),
        )

        ncubos_teoricos = (
            np.array([0.5, 0.75]),
            np.array([1.0, 0.0, 0.5, 0.5]),
            np.array([1.0, 0.5, 0.0, 0.5]),
        )

        for nc_teorico, nc_practico in zip(ncubos_teoricos, bipartido.ncubos):
            assert np.array_equal(nc_teorico, nc_practico.data)
