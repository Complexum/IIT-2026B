"""Página HTML multi-sección: tiempos, box-plots, correlación y error relativo."""

from pathlib import Path

from src.tui.analysis.plots.boxplot import _build_box_fig
from src.tui.analysis.plots.common import (
    RESULTADOS_DIR,
    _grid_shape,
    _pick_ref,
    _strats,
)
from src.tui.analysis.plots.correlation import _build_corr_fig
from src.tui.analysis.plots.time import _build_time_fig
from src.tui.analysis.plots.waterfall import _build_waterfall_fig


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Análisis comparativo · {ref}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: system-ui, sans-serif; }}
  nav {{
    position: sticky; top: 0; z-index: 100;
    background: #16213e; border-bottom: 1px solid #0f3460;
    padding: 10px 24px; display: flex; gap: 24px; align-items: center;
  }}
  nav span {{ color: #64b5f6; font-weight: 600; font-size: 0.95rem; }}
  nav a {{ color: #90caf9; text-decoration: none; font-size: 0.9rem; }}
  nav a:hover {{ color: #fff; }}
  .section {{ padding: 32px 24px 8px; }}
  .section h2 {{
    color: #90caf9; font-size: 1.1rem; font-weight: 500;
    border-bottom: 1px solid #0f3460; padding-bottom: 8px; margin-bottom: 16px;
  }}
  .section p.meta {{ font-size: 0.8rem; color: #888; margin-bottom: 8px; }}
</style>
</head>
<body>
<nav>
  <span>Análisis · ref: {ref}</span>
  <a href="#tiempos">Tiempos</a>
  <a href="#boxplots">Box-plots</a>
  <a href="#correlacion">Correlación</a>
  <a href="#waterfall">Δt vs {ref}</a>
</nav>

<div class="section" id="tiempos">
  <h2>Tiempos de ejecución (escala log)</h2>
  <p class="meta">Una línea por estrategia · eje Y logarítmico · línea punteada = promedio</p>
  {time_html}
</div>

<div class="section" id="boxplots">
  <h2>Distribución de tiempos por estrategia</h2>
  <p class="meta">Un box-plot por estrategia · eje Y logarítmico · línea discontinua = media</p>
  {box_html}
</div>

<div class="section" id="correlacion">
  <h2>Correlación vs {ref}</h2>
  <p class="meta">Scatter Φ_strat vs Φ_ref · verde = dentro de tol · rojo = fuera</p>
  {corr_html}
</div>

<div class="section" id="waterfall">
  <h2>Δt vs {ref} (s)</h2>
  <p class="meta">Diferencia de tiempo vs baseline · positivo = más lento · negativo = más rápido</p>
  {waterfall_html}
</div>

</body>
</html>
"""


def generate_analysis_page(
    groups: dict[tuple[str, str], list[str]],
    reference: str | None = None,
    tol: float = 1e-4,
) -> Path:
    """Genera HTML multi-sección: tiempos, box-plots, error relativo, correlación."""
    from src.paper.loader import load_all_groups

    all_names = [n for names in groups.values() for n in names]
    groups_data = load_all_groups(all_names)
    if not groups_data:
        raise ValueError("Sin datos para graficar")

    # Determinar referencia real para el título
    first_df = next(iter(groups_data.values()))
    ref_used = _pick_ref(_strats(first_df), reference)

    n = len(groups_data)
    rows, cols = _grid_shape(n)

    time_fig = _build_time_fig(groups_data, rows, cols)
    box_fig = _build_box_fig(groups_data, rows, cols)
    waterfall_fig = _build_waterfall_fig(groups_data, reference, rows, cols)
    corr_fig = _build_corr_fig(groups_data, reference, rows, cols, tol)

    time_html = time_fig.to_html(full_html=False, include_plotlyjs="cdn")
    box_html = box_fig.to_html(full_html=False, include_plotlyjs=False)
    waterfall_html = waterfall_fig.to_html(full_html=False, include_plotlyjs=False)
    corr_html = corr_fig.to_html(full_html=False, include_plotlyjs=False)

    html = _HTML_TEMPLATE.format(
        ref=ref_used,
        time_html=time_html,
        box_html=box_html,
        waterfall_html=waterfall_html,
        corr_html=corr_html,
    )

    out_path = RESULTADOS_DIR / "compare_plot.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path
