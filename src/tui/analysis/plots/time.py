"""Figura de tiempos de ejecución por índice (escala log), una línea por estrategia."""

from src.tui.analysis.plots.common import _COLORS, _strats, _subplot_titles


def _build_time_fig(groups_data: dict, rows: int, cols: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=_subplot_titles(groups_data),
        horizontal_spacing=0.08, vertical_spacing=0.20,
    )
    for g_idx, ((ds, pat), df) in enumerate(groups_data.items()):
        r, c = g_idx // cols + 1, g_idx % cols + 1
        idx = df["indice"].to_list()
        for ci, strat in enumerate(sorted(_strats(df))):
            col_t = f"{strat}_t"
            if col_t not in df.columns:
                continue
            vals = df[col_t].to_list()
            mean_t = df[col_t].mean()
            color = _COLORS[ci % len(_COLORS)]
            fig.add_trace(go.Scatter(
                x=idx, y=vals, mode="lines+markers", name=strat,
                legendgroup=strat, showlegend=(g_idx == 0),
                line=dict(color=color, width=2), marker=dict(size=4),
                hovertemplate=f"<b>{strat}</b><br>idx=%{{x}}<br>t=%{{y:.4f}}s<extra></extra>",
            ), row=r, col=c)
            fig.add_hline(y=mean_t, line_dash="dash", line_color=color,
                          annotation_text=f"avg={mean_t:.3f}", row=r, col=c)
        fig.update_xaxes(title_text="índice", row=r, col=c)
        fig.update_yaxes(title_text="tiempo (s)", type="log", row=r, col=c)
    fig.update_layout(
        title_text="Tiempos de ejecución (escala log)",
        height=max(420, 400 * rows), template="plotly_dark",
        margin=dict(t=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
    )
    return fig
