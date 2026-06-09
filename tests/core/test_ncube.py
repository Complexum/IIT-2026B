"""Tests for NCube operations: condicionar, marginalizar, value_at."""

import numpy as np

from src.iit.core.ncube import NCube


class TestNCubeOperations:
    """Test 1.5: NCube operations match Rust output."""

    def test_ncube_creation(self):
        ncubo = NCube(
            indice=0,
            dims=(0, 1, 2),
            data=np.array([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        )
        assert ncubo.indice == 0
        assert ncubo.dims == (0, 1, 2)
        assert len(ncubo.data) == 8

    def test_en(self):
        ncubo = NCube(indice=0, dims=(0, 1, 2), data=np.array([1.0] * 8))

        print(ncubo)

        assert ncubo.idx(0) == 0
        assert ncubo.idx(1) == 1
        assert ncubo.idx(2) == 2
        assert ncubo.idx(3) is None

    def test_value_at(self):
        ncubo = NCube(
            indice=0,
            dims=(0, 1, 2),
            data=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        )

        print(ncubo)

        assert ncubo[(0, 0, 0)] == 0.0
        assert ncubo[(0, 0, 1)] == 0.4
        assert ncubo[(1, 1, 1)] == 0.7

    def test_condicionar(self):
        ncubo = NCube(
            indice=0,
            dims=(0, 1, 2),
            data=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        )
        condicionado = ncubo.condicionar(indices=(2,), estado=(1, 0, 1))
        assert condicionado.indice == 0
        assert 2 not in condicionado.dims
        assert len(condicionado.dims) == 2

    def test_marginalizar(self):
        ncubo = NCube(
            indice=0,
            dims=(0, 1, 2),
            data=np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        )
        marginalizado = ncubo.marginalizar(ejes=(2,))
        assert marginalizado.indice == 0
        assert 2 not in marginalizado.dims
        assert len(marginalizado.dims) == 2
        assert len(marginalizado.data) == 4
