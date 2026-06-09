"""Figura de distribución de tiempos por estrategia (box-plots, escala log)."""

from src.tui.analysis.plots.common import _COLORS, _strats, _subplot_titles


def _build_box_fig(groups_data: dict, rows: int, cols: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=_subplot_titles(groups_data),
        horizontal_spacing=0.08, vertical_spacing=0.20,
    )
    for g_idx, ((ds, pat), df) in enumerate(groups_data.items()):
        r, c = g_idx // cols + 1, g_idx % cols + 1
        strats = _strats(df)
        # Color estable por nombre de estrategia (no por posición)
        color_map = {s: _COLORS[i % len(_COLORS)] for i, s in enumerate(sorted(strats))}
        # Mediana de tiempo por estrategia (descartando las que no tienen columna)
        vals_map = {
            s: df[f"{s}_t"].to_list()
            for s in strats
            if f"{s}_t" in df.columns
        }
        medians = {s: df[f"{s}_t"].median() for s in vals_map}
        # Orden descendente por mediana de tiempo (mayor a la izquierda)
        by_median = sorted(vals_map, key=lambda s: medians[s], reverse=True)
        for strat in by_median:
            color = color_map[strat]
            fig.add_trace(go.Box(
                y=vals_map[strat], name=strat, legendgroup=strat,
                showlegend=(g_idx == 0),
                marker_color=color, line=dict(color=color),
                boxmean=True, boxpoints="outliers",
                hovertemplate=f"<b>{strat}</b><br>t=%{{y:.4f}}s<extra></extra>",
            ), row=r, col=c)
        fig.update_xaxes(title_text="estrategia", row=r, col=c)
        fig.update_yaxes(title_text="tiempo (s)", type="log", row=r, col=c)
    fig.update_layout(
        title_text="Distribución de tiempos por estrategia",
        height=max(420, 400 * rows), template="plotly_dark",
        margin=dict(t=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
    )
    return fig
