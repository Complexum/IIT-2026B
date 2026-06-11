"""Figura de uso de recursos por índice (cpu_user_s, cpu_sys_s, mem_rss_mb, gpu_mem_mb)."""

from src.tui.analysis.plots.common import _COLORS, _strats, _subplot_titles


def _build_resource_fig(groups_data: dict, rows: int, cols: int, metric: str, metric_label: str):
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
            col = f"{strat}_{metric}"
            if col not in df.columns:
                continue
            vals = df[col].to_list()
            mean_val = df[col].mean()
            color = _COLORS[ci % len(_COLORS)]
            fig.add_trace(go.Scatter(
                x=idx, y=vals, mode="lines+markers", name=strat,
                legendgroup=strat, showlegend=(g_idx == 0),
                line=dict(color=color, width=2), marker=dict(size=4),
                hovertemplate=f"<b>{strat}</b><br>idx=%{{x}}<br>{metric_label}=%{{y:.4f}}<extra></extra>",
            ), row=r, col=c)
            fig.add_hline(y=mean_val, line_dash="dash", line_color=color,
                          annotation_text=f"avg={mean_val:.3f}", row=r, col=c)
        fig.update_xaxes(title_text="índice", row=r, col=c)
        fig.update_yaxes(title_text=metric_label, row=r, col=c)
    fig.update_layout(
        title_text=f"{metric_label} por índice",
        height=max(420, 400 * rows), template="plotly_dark",
        margin=dict(t=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
    )
    return fig