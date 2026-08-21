"""Tests de la estrategia QSW (Stoer-Wagner × Queyranne)."""

import numpy as np
import pytest

from src.iit.core.params import Params
from src.iit.strategies.python.qn.oracle import (
    f_cara,
    f_cara_batch,
    preparar_oraculo,
)
from src.iit.strategies.python.qsw.core import (
    OraculoCache,
    seed_weights,
    stoer_wagner_queyranne,
)
from src.iit.strategies.runner import ejecutar
from src.io.manager import cargar_mpt, reducir_a_subsistema

RED = "N15A"

# Dos triángulos unidos por un puente — mismo caso del port crudo (reference.py).
ARISTAS = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3)]


def _grafo(n, aristas):
    w = np.zeros((n, n))
    for u, v in aristas:
        w[u, v] += 1.0
        w[v, u] += 1.0
    return w


def _f_corte(w):
    """f(S) = peso de las aristas que cruzan S ↔ complemento. Simétrica, f(∅)=0."""
    n = w.shape[0]

    def f_batch(masks):
        out = []
        for m in masks:
            dentro = np.array([(m >> u) & 1 for u in range(n)], dtype=bool)
            out.append(w[np.ix_(dentro, ~dentro)].sum())
        return np.array(out, dtype=np.float64)

    return f_batch


class TestNucleoSobreGrafo:
    """Con f gráfica el surrogate W es EXACTO — valida el puente SW ↔ Queyranne."""

    def test_seed_recupera_los_pesos_del_grafo(self):
        w = _grafo(6, ARISTAS)
        W, _ = seed_weights(6, OraculoCache(_f_corte(w)))
        np.testing.assert_allclose(W, w, atol=1e-12)

    @pytest.mark.parametrize("modo", ["exacto", "estatico"])
    def test_encuentra_el_corte_minimo(self, modo):
        w = _grafo(6, ARISTAS)
        valor, mask, _ = stoer_wagner_queyranne(6, _f_corte(w), modo=modo)
        assert valor == pytest.approx(1.0)
        lado = {u for u in range(6) if (mask >> u) & 1}
        assert lado in ({0, 1, 2}, {3, 4, 5})

    @pytest.mark.parametrize("modo", ["exacto", "estatico"])
    def test_conteo_de_oraculo_es_cuadratico(self, modo):
        """La BÚSQUEDA usa O(V²) llamadas, no O(V³): es lo que justifica el híbrido.

        Se mide `generar_candidatas` sola. La reparación posterior (1-opt y 2-opt)
        es un post-pass aparte y crece como O(V⁴); mezclarla acá escondería la
        propiedad que el híbrido realmente aporta.
        """
        from src.iit.strategies.python.qsw.core import OraculoCache, generar_candidatas

        V = 12
        rng = np.random.default_rng(0)
        w = rng.random((V, V))
        w = np.triu(w, 1)
        w += w.T
        orc = OraculoCache(_f_corte(w))
        generar_candidatas(V, orc, modo=modo)
        assert orc.llamadas < V**3 / 4
        if modo == "estatico":
            # Todo el oráculo de la búsqueda en un único batch (+ el seed).
            assert orc.batches <= 2

    def test_mascara_nunca_es_trivial(self):
        w = _grafo(6, ARISTAS)
        _, mask, _ = stoer_wagner_queyranne(6, _f_corte(w))
        assert mask not in (0, (1 << 6) - 1)


def _subsistema():
    tpm = cargar_mpt(RED)
    n = tpm.shape[1]
    return reducir_a_subsistema(tpm, Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n))


class TestOraculo:
    def test_f_cara_batch_identico_a_f_cara(self):
        """Condición para no romper qn / qn_mul / qn_cuda."""
        sistema = _subsistema()
        orc = preparar_oraculo(sistema)
        dims, indices = sistema.dims, sistema.indices
        rng = np.random.default_rng(7)

        cortes = []
        for _ in range(24):
            mec = tuple(d for d in dims if rng.random() < 0.5)
            alc = tuple(i for i in indices if rng.random() < 0.5)
            cortes.append((alc, mec))

        esperado = np.array([f_cara(orc, a, m) for a, m in cortes])

        masks = np.array(
            [sum(1 << orc.pos_dim[d] for d in m) for _, m in cortes], dtype=np.int64
        )
        alc_bool = np.array(
            [[i in set(a) for i in indices] for a, _ in cortes], dtype=bool
        )
        obtenido = f_cara_batch(orc, masks, alc_bool)

        np.testing.assert_array_equal(obtenido.astype(np.float64), esperado)

    def test_f_es_simetrica_y_anclada(self):
        """f(∅) = f(V) = 0 y f(S) = f(V∖S): requisito de Queyranne/Stoer-Wagner."""
        sistema = _subsistema()
        orc = preparar_oraculo(sistema)
        dims, indices = sistema.dims, sistema.indices

        assert f_cara(orc, (), ()) == pytest.approx(0.0, abs=1e-5)
        assert f_cara(orc, indices, dims) == pytest.approx(0.0, abs=1e-5)

        rng = np.random.default_rng(11)
        for _ in range(16):
            mec = tuple(d for d in dims if rng.random() < 0.5)
            alc = tuple(i for i in indices if rng.random() < 0.5)
            comp_mec = tuple(d for d in dims if d not in mec)
            comp_alc = tuple(i for i in indices if i not in alc)
            assert f_cara(orc, alc, mec) == pytest.approx(
                f_cara(orc, comp_alc, comp_mec), rel=1e-5, abs=1e-6
            )


class TestEstrategia:
    @pytest.mark.parametrize("modo", ["exacto", "estatico", "estocastico"])
    def test_registrada_y_ejecutable(self, modo):
        import math

        tpm = cargar_mpt(RED)
        n = tpm.shape[1]
        sol = ejecutar(
            tpm,
            Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n),
            "qsw",
            opciones={"modo": modo},
        )
        assert math.isfinite(sol.perdida)
        assert sol.perdida >= 0.0
        assert sol.particion != ""

    def test_no_peor_que_analytic_por_mucho(self):
        """`analytic` es exacto: qsw nunca debe quedar por debajo de él."""
        tpm = cargar_mpt(RED)
        n = tpm.shape[1]
        params = Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)
        ref = ejecutar(tpm, params, "analytic").perdida
        got = ejecutar(tpm, params, "qsw").perdida
        assert got >= ref - 1e-6


class TestRegistroDeEstrategias:
    """El rename swq→qsw rompió la ejecución sin que ningún test lo notara.

    `listar_estrategias()` descubre estrategias por **nombre de carpeta** (es lo
    que puebla el dropdown del tab Execution y lo que valida `cli run`), pero
    `runner.ejecutar` resuelve contra `SIA.registry`. Si los dos conjuntos no
    coinciden, la estrategia aparece en la UI y revienta al ejecutarla.
    """

    @staticmethod
    def _conjuntos():
        import importlib
        from pathlib import Path

        from src.iit.strategies.python.sia import SIA
        from src.tui.run.helpers import listar_estrategias

        for d in sorted(Path("src/iit/strategies/python").iterdir()):
            if d.is_dir() and (d / "code.py").exists():
                importlib.import_module(f"src.iit.strategies.python.{d.name}.code")
        return set(listar_estrategias()), set(SIA.registry)

    def test_toda_carpeta_registra_su_nombre(self):
        carpetas, registro = self._conjuntos()
        assert not (carpetas - registro), (
            f"carpetas visibles en la UI pero no registradas: {sorted(carpetas - registro)}"
        )

    def test_toda_estrategia_registrada_tiene_carpeta(self):
        carpetas, registro = self._conjuntos()
        assert not (registro - carpetas), (
            f"registradas pero invisibles para la UI / `cli run`: {sorted(registro - carpetas)}"
        )

    def test_qsw_presente(self):
        _, registro = self._conjuntos()
        assert "qsw" in registro


class TestOpciones:
    """`modo` y `backend` son atributos declarados en `QSW.opciones`."""

    def test_defaults(self):
        from src.iit.strategies.python.qsw.code import QSW

        assert QSW.defaults() == {"modo": "exacto", "backend": "python", "k": "auto"}

    @pytest.mark.parametrize(
        "opciones, mensaje",
        [
            ({"backend": "rust"}, "inválido"),
            ({"modo": "turbo"}, "inválido"),
            ({"inexistente": "x"}, "no admite la opción"),
        ],
    )
    def test_opcion_invalida_lanza_sin_instanciar(self, opciones, mensaje):
        """Debe fallar antes de arrancar, no correr con el default en silencio."""
        from src.iit.strategies.python.qsw.code import QSW

        with pytest.raises(ValueError, match=mensaje):
            QSW.validar_opciones(opciones)

    def test_backend_c_falla_explicito_sin_libreria(self):
        """Nunca degradar a Python en silencio: la opción entra en el nombre del
        CSV, y un resultado etiquetado `backend=c` que corrió Python mentiría."""
        from src.iit.strategies.python.qsw.backend import (
            cargar_libqsw,
            resolver_backend,
        )

        if cargar_libqsw() is not None:
            pytest.skip("libqsw.so compilada — el backend C sí está disponible")
        with pytest.raises(RuntimeError, match="backend 'c' no disponible"):
            resolver_backend("c")
        assert resolver_backend("auto") == "python"
        assert resolver_backend("python") == "python"

    def test_etiqueta_solo_incluye_lo_no_default(self):
        """El nombre del CSV distingue corridas con opciones distintas."""
        from src.tui.run.helpers import etiqueta_estrategia

        assert etiqueta_estrategia("qsw", None) == "qsw"
        assert etiqueta_estrategia("qsw", {"modo": "exacto"}) == "qsw"
        assert etiqueta_estrategia("qsw", {"modo": "estatico"}) == "qsw+modo=estatico"


class TestMuestreo:
    """`modo=estocastico`: nunca construye la tabla de 2^D caras."""

    def test_k_escala_con_la_red(self):
        """K fijo y chico falla al crecer la red (n=16/K=256 erró 68 % en el
        prototipo). Por eso K crece con N en vez de quedarse quieto."""
        from src.iit.strategies.python.qsw.muestreo import K_BASE, k_efectivo

        assert k_efectivo(4) == K_BASE                 # redes chicas: piso
        assert k_efectivo(64) > k_efectivo(8)          # crece con N
        assert k_efectivo(10, "8192") == 8192          # override explícito

    def test_caras_chicas_son_exactas(self):
        """Muestrear más puntos que celdas tiene no aporta nada y es donde vive
        el error: si 2^|A| <= K la cara se suma entera."""
        from src.iit.strategies.python.qsw.muestreo import OraculoMuestreado

        sistema = _subsistema()
        orc = OraculoMuestreado(sistema, k="1024")
        D = len(sistema.dims)
        orc.f_batch([1, 3, (1 << D) - 1])
        assert orc.caras_exactas > 0

    def test_no_materializa_las_2_a_la_D_caras(self):
        """La tabla Zeta es (N, 2^D); el muestreo sólo guarda O(D²) medias."""
        from src.iit.strategies.python.qsw.muestreo import OraculoMuestreado

        sistema = _subsistema()
        D, N = len(sistema.dims), len(sistema.indices)
        orc = OraculoMuestreado(sistema)
        orc.f_batch([1 << u for u in range(D + N)])
        assert len(orc._cache) <= 2 * (D + N) + 2

    def test_reporta_confianza_en_vez_de_afirmar_exactitud(self):
        from src.iit.strategies.python.qsw.muestreo import OraculoMuestreado

        sistema = _subsistema()
        D, N = len(sistema.dims), len(sistema.indices)
        orc = OraculoMuestreado(sistema)
        info = orc.confianza([1 << u for u in range(D + N)])
        assert set(info) == {"mask", "valor", "margen", "ruido", "confiable"}
        assert info["ruido"] >= 0.0

    def test_coincide_con_analytic(self):
        tpm = cargar_mpt(RED)
        n = tpm.shape[1]
        params = Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)
        ref = ejecutar(tpm, params, "analytic").perdida
        got = ejecutar(tpm, params, "qsw", opciones={"modo": "estocastico"}).perdida
        assert abs(got - ref) < 1e-6


class TestMedicionDeTiempo:
    """El CSV debe medir el algoritmo, no la preparación del subsistema.

    Antes el `ResourceMonitor` envolvía `ejecutar()`, que incluye
    `reducir_a_subsistema`. Como esa preparación cuesta lo mismo para todas las
    estrategias, actuaba como constante compartida y comprimía los speedups hacia
    1: medido sobre N20A, `qsw+backend=c` aparecía 1.55× cuando el algoritmo va
    3.25×.
    """

    def test_monitor_envuelve_solo_el_algoritmo(self):
        """El reloj externo y el que la estrategia se mide sola deben coincidir.

        Si difieren mucho, el monitor volvió a envolver la preparación.
        """
        import time

        from src.iit.strategies.runner import preparar_subsistema, resolver_estrategia

        tpm = cargar_mpt(RED)
        n = tpm.shape[1]
        params = Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)

        sub = preparar_subsistema(tpm, params)
        t0 = time.perf_counter()
        sol = resolver_estrategia(sub, "qsw")
        externo = time.perf_counter() - t0

        assert sol.tiempo_ejecucion == pytest.approx(externo, abs=5e-3)

    def test_preparar_y_resolver_equivalen_a_ejecutar(self):
        from src.iit.strategies.runner import preparar_subsistema, resolver_estrategia

        tpm = cargar_mpt(RED)
        n = tpm.shape[1]
        params = Params("1" + "0" * (n - 1), "1" * n, "1" * n, "1" * n)

        compuesto = ejecutar(tpm, params, "qsw").perdida
        separado = resolver_estrategia(preparar_subsistema(tpm, params), "qsw").perdida
        assert compuesto == separado

    def test_csv_trae_la_columna_de_preparacion(self):
        from src.tui.run.csv_utils import CSV_HEADERS

        assert "tiempo_preparacion_s" in CSV_HEADERS
        # `tiempo_wall_s` se conserva: los nueve consumidores lo leen y ahora
        # significa lo correcto, así que ninguno necesitó cambiar.
        assert "tiempo_wall_s" in CSV_HEADERS

    def test_detecta_formatos_mezclados(self):
        """CSV viejos (sin la columna) miden otra cosa: hay que avisar, no callar."""
        from src.tui.analysis.compare import formato_nuevo, formatos_mezclados

        assert formatos_mezclados([]) == []
        # Un conjunto homogéneo no dispara aviso, sea del formato que sea.
        assert formatos_mezclados(["no/existe/a", "no/existe/b"]) == []
        assert formato_nuevo("no/existe") is False


class TestOpcionesPorEstrategia:
    """Las opciones son de cada estrategia; al cambiarla, las viejas no aplican.

    Un programa que fue `qsw` con `backend=c` y pasó a `queyranne` fallaba con
    *"'queyranne' no admite la opción 'backend'"* — y como `backend` no lo declara
    ninguna otra estrategia, rompía con cualquiera que se eligiera.
    """

    def test_poda_lo_que_la_nueva_estrategia_no_admite(self):
        from src.tui.run.helpers import podar_opciones

        validas, descartadas = podar_opciones("queyranne", {"backend": "c"})
        assert validas == {}
        assert descartadas == {"backend": "c"}

    def test_conserva_lo_que_si_admite(self):
        from src.tui.run.helpers import podar_opciones

        validas, descartadas = podar_opciones(
            "qsw", {"backend": "c", "modo": "estatico"}
        )
        assert validas == {"backend": "c", "modo": "estatico"}
        assert descartadas == {}

    def test_poda_valores_invalidos_aunque_el_atributo_exista(self):
        from src.tui.run.helpers import podar_opciones

        validas, descartadas = podar_opciones("qsw", {"backend": "rust"})
        assert validas == {}
        assert descartadas == {"backend": "rust"}

    def test_estrategia_desconocida_no_poda(self):
        """Que se queje quien ejecute, con su propio mensaje."""
        from src.tui.run.helpers import podar_opciones

        validas, _ = podar_opciones("no_existe", {"backend": "c"})
        assert validas == {"backend": "c"}

    def test_mensaje_util_cuando_no_hay_opciones(self):
        from src.iit.strategies.python.sia import SIA
        from src.iit.strategies.runner import importar_estrategias

        importar_estrategias()
        with pytest.raises(ValueError, match="no tiene opciones configurables"):
            SIA.registry["queyranne"].validar_opciones({"backend": "python"})
