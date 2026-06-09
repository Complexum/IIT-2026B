"""Tests for IO layer: list, load, build, delete networks."""

from pathlib import Path

import numpy as np
import pytest

from src.io.generator import peso_estimado, generar_red
from src.io.manager import (
    crear_sistema,
    eliminar_red,
    listar_redes,
    listar_resultados,
    cargar_resultados,
    cargar_mpt,
)

NETWORKS_DIR = Path("data/input/networks")


class TestListNetworks:
    """Test 1.1: List all available networks."""

    def test_list_networks_returns_list(self):
        networks = listar_redes()
        assert isinstance(networks, list)
        assert all(isinstance(n, str) for n in networks)

    def test_list_networks_contains_known_networks(self):
        networks = listar_redes()
        assert "N1A" in networks
        assert "N2A" in networks
        assert "N3A" in networks


class TestLoadTPM:
    """Test 1.2: Load TPM from CSV."""

    def test_load_n1a_has_correct_shape(self):
        tpm = cargar_mpt("N1A")
        print(tpm)
        assert tpm.shape == (1, 2)
        assert isinstance(tpm, np.ndarray)

    def test_load_n2a_has_correct_shape(self):
        tpm = cargar_mpt("N2A")
        print(tpm)
        assert tpm.shape == (4, 2)
        assert isinstance(tpm, np.ndarray)

    def test_load_n3a_has_correct_shape(self):
        tpm = cargar_mpt("N3A")
        assert tpm.shape == (8, 3)
        assert isinstance(tpm, np.ndarray)

    def test_load_nonexistent_raises_error(self):
        with pytest.raises(FileNotFoundError):
            cargar_mpt("NONEXISTENT")


class TestBuildSystem:
    """Test 1.3: Build System from TPM."""

    def test_build_system_from_n3a(self):
        tpm = cargar_mpt("N3A")
        estado = (1, 0, 0)
        sistema = crear_sistema(tpm, estado)

        assert sistema.estado_inicial == estado
        assert len(sistema.ncubos) == 3
        assert sistema.indices == (0, 1, 2)
        assert sistema.dims == (0, 1, 2)

    def test_build_system_ncubes_have_correct_data(self):
        tpm = cargar_mpt("N3A")
        estado = (0, 0, 0)
        sistema = crear_sistema(tpm, estado)

        for i, ncubo in enumerate(sistema.ncubos):
            assert ncubo.indice == i
            assert ncubo.dims == (0, 1, 2)
            np.testing.assert_array_equal(ncubo.data, tpm[:, i])


class TestGenerator:
    """Test 1.4: Generate networks."""

    def test_estimate_size_calculates_correctly(self):
        size_3 = peso_estimado(3)
        assert size_3 > 0
        assert size_3 < 1  # 8*3 bytes = 24 bytes, muy pequeño

        size_10 = peso_estimado(10)
        assert size_10 > size_3

    def test_generar_red_creates_file(self):
        filename = generar_red(2, datos_deterministas=True)
        assert filename.startswith("N2")
        assert filename.endswith(".csv")

        path = NETWORKS_DIR / filename
        assert path.exists()

        # Cleanup
        path.unlink()

    def test_generar_red_deterministic_has_binary_values(self):
        filename = generar_red(2, datos_deterministas=True)
        tpm = cargar_mpt(filename.replace(".csv", ""))

        assert set(tpm.flatten()).issubset({0.0, 1.0})

        # Cleanup
        (NETWORKS_DIR / filename).unlink()

    def test_generar_red_stochastic_has_continuous_values(self):
        filename = generar_red(2, datos_deterministas=False)
        tpm = cargar_mpt(filename.replace(".csv", ""))

        assert not set(tpm.flatten()).issubset({0.0, 1.0})
        assert np.all((tpm >= 0) & (tpm <= 1))

        # Cleanup
        (NETWORKS_DIR / filename).unlink()


class TestDeleteNetwork:
    """Test delete network functionality."""

    def test_delete_network_removes_file(self):
        # Create a test network
        filename = generar_red(2, datos_deterministas=True)
        name = filename.replace(".csv", "")

        assert (NETWORKS_DIR / filename).exists()

        # Delete it
        deleted = eliminar_red(name)
        assert deleted is True
        assert not (NETWORKS_DIR / filename).exists()

    def test_delete_nonexistent_returns_false(self):
        deleted = eliminar_red("NONEXISTENT")
        assert deleted is False


class TestResults:
    """Test results loading."""

    def test_list_results_returns_list(self):
        results = listar_resultados()
        assert isinstance(results, list)

    def test_load_result_returns_string(self):
        results = listar_resultados()
        if results:
            content = cargar_resultados(results[0])
            assert isinstance(content, str)
            assert len(content) > 0
