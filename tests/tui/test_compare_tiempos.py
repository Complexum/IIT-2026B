"""Tests del resumen de tiempos de `cli compare`.

`compare_group_n` responde si dos estrategias *coinciden*; estas funciones
responden *cuál conviene*. Los tests fijan el contrato de la distribución
(cinco números + bigotes de Tukey) y el de la línea de veredicto.
"""

import polars as pl
import pytest

from src.tui.analysis.compare import (
    build_rich_table_tiempos,
    times_summary,
    stamps_veredict,
)


def _merged(**columnas):
    """DataFrame con la forma que devuelve `compare_group_n`: una `<estrategia>_t`
    por estrategia más `indice`."""
    n = len(next(iter(columnas.values())))
    datos = {"indice": list(range(n))}
    datos.update({f"{k}_t": [float(v) for v in vals] for k, vals in columnas.items()})
    return pl.DataFrame(datos)


class TestResumenTiempos:
    def test_cinco_numeros_conocidos(self):
        # 1..9 → Q1=3, mediana=5, Q3=7, sin atípicos (bigotes = min y max).
        r = times_summary(_merged(a=range(1, 10)), ["a"])["a"]
        assert r["n"] == 9
        assert r["min"] == 1.0
        assert r["q1"] == 3.0
        assert r["mediana"] == 5.0
        assert r["media"] == 5.0
        assert r["q3"] == 7.0
        assert r["max"] == 9.0
        assert r["bigote_inf"] == 1.0
        assert r["bigote_sup"] == 9.0
        assert r["atipicos"] == 0
        assert r["total"] == 45.0

    def test_el_bigote_recorta_el_atipico_pero_el_max_lo_conserva(self):
        # El bigote es el dato más extremo DENTRO de la cota, no la cota misma.
        r = times_summary(_merged(a=[1] * 8 + [100]), ["a"])["a"]
        assert r["max"] == 100.0
        assert r["bigote_sup"] == 1.0
        assert r["atipicos"] == 1

    def test_media_y_mediana_se_separan_con_cola(self):
        # El caso real: barridos con D chico y D grande mezclados. Si sólo se
        # reportara la media, la cola la arrastraría y parecería el caso típico.
        r = times_summary(_merged(a=[1] * 9 + [1000]), ["a"])["a"]
        assert r["mediana"] == 1.0
        assert r["media"] > 100

    def test_todo_igual_no_rompe_con_iqr_cero(self):
        r = times_summary(_merged(a=[2, 2, 2, 2]), ["a"])["a"]
        assert r["bigote_inf"] == r["bigote_sup"] == 2.0
        assert r["atipicos"] == 0

    def test_ignora_estrategias_sin_columna_o_vacias(self):
        assert times_summary(_merged(a=[1.0]), ["a", "fantasma"]).keys() == {"a"}
        assert times_summary(pl.DataFrame({"indice": []}), ["a"]) == {}

    def test_varias_estrategias(self):
        r = times_summary(_merged(a=[1, 2, 3], b=[10, 20, 30]), ["a", "b"])
        assert r["a"]["mediana"] == 2.0
        assert r["b"]["mediana"] == 20.0


class TestVeredicto:
    def test_reporta_mediana_y_total_aunque_gane_la_misma(self):
        # La brecha entre los dos factores es el dato: dice si la ventaja está en
        # el caso típico o en la cola.
        r = times_summary(_merged(a=[1, 1, 1, 1], b=[1, 1, 1, 100]), ["a", "b"])
        texto = stamps_veredict(r)
        assert "1.00x por mediana" in texto
        assert "25.75x por tiempo total" in texto

    def test_avisa_cuando_los_ganadores_diferen(self):
        # `a` gana la mediana, `b` gana el total.
        r = times_summary(_merged(a=[1, 1, 1, 100], b=[9, 9, 9, 9]), ["a", "b"])
        texto = stamps_veredict(r)
        assert "caso típico" in texto and "cola" in texto

    def test_una_sola_estrategia_no_tiene_veredicto(self):
        assert stamps_veredict(times_summary(_merged(a=[1.0]), ["a"])) == ""


class TestTabla:
    def test_la_mejor_mediana_va_primero(self):
        r = times_summary(_merged(lenta=[10, 10, 10], rapida=[1, 1, 1]),
                            ["lenta", "rapida"])
        tabla = build_rich_table_tiempos(r)
        assert [c.header for c in tabla.columns] == ["", "rapida", "lenta"]

    def test_tiene_las_metricas_pedidas(self):
        r = times_summary(_merged(a=[1, 2, 3]), ["a"])
        etiquetas = list(build_rich_table_tiempos(r).columns[0]._cells)
        assert etiquetas == [
            "n", "max", "bigote ↑", "Q3", "media", "mediana", "Q1",
            "bigote ↓", "min", "atípicos", "total", "vs mejor",
        ]

    def test_las_filas_van_de_mayor_a_menor_como_un_boxplot(self):
        r = times_summary(_merged(a=range(1, 10)), ["a"])
        valores = [c for c in build_rich_table_tiempos(r).columns[1]._cells[1:9]]
        numeros = [float(v) for v in valores]
        assert numeros == sorted(numeros, reverse=True)

    def test_resumen_vacio_no_rompe(self):
        build_rich_table_tiempos({})
