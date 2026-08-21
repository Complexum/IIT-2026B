"""Diagnóstico de identificabilidad (`src/io/identificabilidad.py`).

Antes de intentar reconstruir una spec compacta desde una TPM hay que saber si es
posible. Dos maneras de que no lo sea, y hay que distinguirlas: **densa** (no hay
compresión, por conteo) y **bajo el ruido** (la TPM es una estimación y las
dependencias son menores que su propio error).
"""

import numpy as np
import pytest

from src.io.fst import generar_espec, generar_espec_pbn, tpm_nodo_estado
from src.io.identificabilidad import diagnosticar, influencias, piso_de_ruido


class TestInfluencias:
    def test_recupera_los_padres_de_una_red_generada(self):
        """Influencia exactamente 0 en las dims que no son padres del nodo."""
        espec = generar_espec(6, 2, semilla=7)
        infl = influencias(tpm_nodo_estado(espec))
        for i in range(espec.n):
            vivas = tuple(np.flatnonzero(infl[i] > 0).tolist())
            assert vivas == espec.padres[i]

    def test_forma_es_nodos_por_dims(self):
        infl = influencias(tpm_nodo_estado(generar_espec(5, 2, semilla=1)))
        assert infl.shape == (5, 5)


class TestPisoDeRuido:
    def test_tpm_determinista_no_tiene_ruido(self):
        piso, muestras = piso_de_ruido(tpm_nodo_estado(generar_espec(5, 2, semilla=1)))
        assert piso == 0.0
        assert muestras is None

    def test_frecuencias_empiricas_dan_el_denominador(self):
        """Una frecuencia de m muestras es múltiplo de 1/m."""
        tpm = np.array([[k / 50.0] for k in range(1, 50)])
        _, muestras = piso_de_ruido(tpm)
        assert muestras is not None and muestras <= 50


class TestVeredicto:
    def test_red_dispersa_es_identificable(self):
        """Control positivo: una red generada con k=2 debe delatar su fan-in."""
        d = diagnosticar(tpm_nodo_estado(generar_espec(8, 2, semilla=7)))
        assert d["veredicto"].startswith("IDENTIFICABLE")
        assert int(d["fan_in_estimado"].max()) == 2

    def test_red_pbn_dispersa_tambien(self):
        d = diagnosticar(tpm_nodo_estado(generar_espec_pbn(8, 2, funciones=3, semilla=7)))
        assert d["veredicto"].startswith("IDENTIFICABLE")
        assert int(d["fan_in_estimado"].max()) == 2

    def test_tpm_aleatoria_densa_no_es_identificable(self):
        """Lo que produce `generator.generar_red`: cada celda sorteada aparte."""
        rng = np.random.default_rng(0)
        n = 8
        tpm = rng.integers(2, size=(1 << n, n)).astype(np.float64)
        d = diagnosticar(tpm)
        assert d["veredicto"].startswith("DENSA")
        assert int(d["fan_in_estimado"].min()) == n

    def test_dependencias_fuertes_sobreviven_al_ruido(self):
        """Un k=2 determinista tiene influencia ~0.5–1.0 sobre sus padres, muy por
        encima del piso de 8 muestras (~0.2): sigue siendo identificable. Por eso el
        caso de N15A es notable — ahí las dependencias son genuinamente **débiles**
        (0.043), no sólo ruidosas."""
        base = tpm_nodo_estado(generar_espec(6, 2, semilla=7))
        rng = np.random.default_rng(3)
        m = 8
        d = diagnosticar(rng.binomial(m, np.clip(base, 0.02, 0.98)) / m)
        assert d["muestras_estimadas"] is not None
        assert d["veredicto"].startswith("IDENTIFICABLE")

    def test_senal_debil_bajo_ruido_no_es_identificable(self):
        """Dependencias débiles MÁS ruido de muestreo: el caso de N15A. Con ε=0.45 la
        tabla vale 0.45/0.55, así que la influencia real es ~0.1 — por debajo del
        piso que dejan 8 muestras por estado. No es lo mismo que ser densa."""
        base = tpm_nodo_estado(generar_espec_pbn(6, 2, ruido=0.45, semilla=7))
        rng = np.random.default_rng(3)
        m = 8
        d = diagnosticar(rng.binomial(m, np.clip(base, 0.02, 0.98)) / m)
        assert d["muestras_estimadas"] is not None
        assert d["veredicto"].startswith("NO IDENTIFICABLE")
