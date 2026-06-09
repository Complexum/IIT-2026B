"""Figura de correlación Φ_estrategia vs Φ_referencia por grupo."""

from src.tui.analysis.plots.common import _pick_ref, _strats, _subplot_titles


def _build_corr_fig(groups_data: dict, reference: str | None, rows: int, cols: int, tol: float):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=_subplot_titles(groups_data),
        horizontal_spacing=0.10, vertical_spacing=0.22,
    )
    ref_label = reference or "ref"
    for g_idx, ((ds, pat), df) in enumerate(groups_data.items()):
        r, c = g_idx // cols + 1, g_idx % cols + 1
        strats = _strats(df)
        ref = _pick_ref(strats, reference)
        ref_vals = df[ref].to_list()
        ci = 0
        for strat in sorted(strats):
            if strat == ref:
                continue
            strat_vals = df[strat].to_list()
            colors_pt = [
                "#2ecc71" if abs(s - rv) <= tol else "#e74c3c"
                for s, rv in zip(strat_vals, ref_vals)
            ]
            fig.add_trace(go.Scatter(
                x=ref_vals, y=strat_vals, mode="markers",
                name=strat, legendgroup=strat,
                showlegend=(g_idx == 0),
                marker=dict(color=colors_pt, size=7, opacity=0.8),
                hovertemplate=(
                    f"<b>{strat}</b><br>{ref}=%{{x:.6f}}<br>"
                    f"{strat}=%{{y:.6f}}<extra></extra>"
                ),
            ), row=r, col=c)
            ci += 1
        # Diagonal y=x
        lo = min(ref_vals) * 0.95 if ref_vals else 0
        hi = max(ref_vals) * 1.05 if ref_vals else 1
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(dash="dash", color="white", width=1),
            showlegend=False,
        ), row=r, col=c)
        fig.update_xaxes(title_text=f"Φ {ref}", row=r, col=c)
        fig.update_yaxes(title_text="Φ estrategia", row=r, col=c)
    fig.update_layout(
        title_text=f"Correlación vs {ref_label}  (🟢 ≤tol · 🔴 >tol)",
        height=max(420, 400 * rows), template="plotly_dark",
        margin=dict(t=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
    )
    return fig
