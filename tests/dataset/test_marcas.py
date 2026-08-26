"""Marcadores de capacidad de una red (`manager.MARCAS`).

Un marcador emoji rompe el layout de la TUI: `rich` cuenta `🅒` (U+1F152) y `Ⓢ`
(U+1F1F8) como **1 celda** y el terminal los pinta con **2**, así que Textual maqueta
con el ancho equivocado y el botón de borrar se sale del panel. `Ⓢ` es además un
*regional indicator*: dos adyacentes se fusionan en la bandera de un país.
"""

import unicodedata

import pytest
from rich.cells import cell_len

from src.io.manager import MARCAS, marcas_de_red

#: Bloques cuyos glifos tienen presentación **emoji** por defecto y por lo tanto
#: ancho ambiguo entre `rich` y el terminal.
RANGOS_EMOJI = (
    (0x1F000, 0x1FAFF),  # pictogramas, alfanuméricos encerrados suplementarios
    (0x1F1E6, 0x1F1FF),  # regional indicators (componentes de bandera)
    (0x2600, 0x27BF),    # misc symbols & dingbats
)


def _es_emoji(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in RANGOS_EMOJI)


class TestGlifos:
    @pytest.mark.parametrize("glifo, clave", MARCAS)
    def test_mide_una_celda(self, glifo, clave):
        assert cell_len(glifo) == 1, (
            f"{clave}: {glifo!r} mide {cell_len(glifo)} celdas para rich; "
            "un marcador tiene que ocupar exactamente una"
        )

    @pytest.mark.parametrize("glifo, clave", MARCAS)
    def test_no_es_emoji(self, glifo, clave):
        for ch in glifo:
            assert not _es_emoji(ch), (
                f"{clave}: {ch!r} (U+{ord(ch):04X}, "
                f"{unicodedata.name(ch, '?')}) tiene presentación emoji — "
                "rich y el terminal no coinciden en su ancho"
            )

    def test_no_hay_regional_indicators(self):
        """Dos adyacentes formarían una bandera."""
        for glifo, clave in MARCAS:
            for ch in glifo:
                assert not (0x1F1E6 <= ord(ch) <= 0x1F1FF), f"{clave}: {ch!r}"

    def test_las_claves_son_las_capacidades(self):
        assert {clave for _, clave in MARCAS} == {"spec", "csv", "npy"}

    def test_los_glifos_son_distintos(self):
        glifos = [g for g, _ in MARCAS]
        assert len(set(glifos)) == len(glifos)


class TestComposicion:
    def test_el_ancho_total_es_la_cantidad_de_marcas(self, tmp_path, monkeypatch):
        import numpy as np

        import src.io.fst as fst
        import src.io.manager as manager

        monkeypatch.setattr(manager, "REDES_DIR", tmp_path)
        monkeypatch.setattr(fst, "REDES_DIR", tmp_path)
        espec = fst.generar_espec(4, 2, topologia="anillo", semilla=5)
        np.savetxt(tmp_path / "N4A.csv", fst.tpm_nodo_estado(espec), delimiter=",", fmt="%d")
        fst.guardar_espec("N4A", espec)

        marcas = marcas_de_red("N4A")
        assert len(marcas) == 2  # spec + csv, sin npy
        assert cell_len(marcas) == 2

    def test_red_inexistente_no_tiene_marcas(self, tmp_path, monkeypatch):
        import src.io.manager as manager

        monkeypatch.setattr(manager, "REDES_DIR", tmp_path)
        assert marcas_de_red("NoExiste") == ""
