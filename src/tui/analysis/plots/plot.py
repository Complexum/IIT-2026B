"""Main de la categoría de gráficos: agrupa y re-exporta la API pública.

Los gráficos viven en submódulos:
- comparison.py  → generate_comparison_plot (comparación de pares, 4 subplots)
- page.py        → generate_analysis_page (página multi-sección)
- time.py        → tiempos de ejecución por índice
- boxplot.py     → distribución de tiempos por estrategia
- waterfall.py   → Δt vs baseline (waterfall)
- correlation.py → correlación Φ_strat vs Φ_ref
- resource.py    → uso de recursos por índice (cpu_user_s, cpu_sys_s, mem_rss_mb, gpu_mem_mb)
- common.py      → helpers compartidos + open_plot
"""

from src.tui.analysis.plots.common import open_plot
from src.tui.analysis.plots.comparison import generate_comparison_plot
from src.tui.analysis.plots.page import generate_analysis_page

__all__ = ["generate_comparison_plot", "generate_analysis_page", "open_plot"]
