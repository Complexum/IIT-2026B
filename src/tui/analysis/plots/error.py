"""Figura de error relativo de cada estrategia respecto a la referencia."""

from src.tui.analysis.plots.common import _COLORS, _pick_ref, _strats, _subplot_titles


def _build_error_fig(groups_data: dict, reference: str | None, rows: int, cols: int, tol: float):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=_subplot_titles(groups_data),
        horizontal_spacing=0.08, vertical_spacing=0.20,
    )
    ref_label = reference or "ref"
    for g_idx, ((ds, pat), df) in enumerate(groups_data.items()):
        r, c = g_idx // cols + 1, g_idx % cols + 1
        strats = _strats(df)
        ref = _pick_ref(strats, reference)
        ref_vals = df[ref].to_list()
        idx = df["indice"].to_list()
        ci = 0
        for strat in sorted(strats):
            if strat == ref:
                continue
            strat_vals = df[strat].to_list()
            rel_err = [
                abs(s - rv) / max(abs(rv), 1e-12) * 100
                for s, rv in zip(strat_vals, ref_vals)
            ]
            color = _COLORS[ci % len(_COLORS)]
            fig.add_trace(go.Scatter(
                x=idx, y=rel_err, mode="lines+markers",
                name=f"{strat} vs {ref}", legendgroup=strat,
                showlegend=(g_idx == 0),
                line=dict(color=color, width=2), marker=dict(size=4),
                hovertemplate=(
                    f"<b>{strat}</b><br>idx=%{{x}}<br>"
                    "err=%{y:.3f}%<extra></extra>"
                ),
            ), row=r, col=c)
            ci += 1
        fig.add_hline(y=tol * 100, line_dash="dot", line_color="orange",
                      annotation_text=f"tol={tol:.0e}", row=r, col=c)
        fig.update_xaxes(title_text="índice", row=r, col=c)
        fig.update_yaxes(title_text="error relativo (%)", row=r, col=c)
    fig.update_layout(
        title_text=f"Error relativo vs {ref_label}",
        height=max(420, 400 * rows), template="plotly_dark",
        margin=dict(t=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
    )
    return fig
