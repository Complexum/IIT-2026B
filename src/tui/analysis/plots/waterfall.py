"""Figura waterfall: Δtiempos respecto a la referencia baseline por cada índice."""

from src.tui.analysis.plots.common import _COLORS, _pick_ref, _strats, _subplot_titles


def _build_waterfall_fig(groups_data: dict, reference: str | None, rows: int, cols: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=_subplot_titles(groups_data),
        horizontal_spacing=0.08, vertical_spacing=0.20,
    )
    ref_label = reference or "ref"

    all_strats = sorted({s for df in groups_data.values() for s in _strats(df)})
    strat_to_ci = {s: i for i, s in enumerate(all_strats)}

    for g_idx, ((ds, pat), df) in enumerate(groups_data.items()):
        r, c = g_idx // cols + 1, g_idx % cols + 1
        strats = _strats(df)
        ref = _pick_ref(strats, reference)
        ref_col_t = f"{ref}_t"
        if ref_col_t not in df.columns:
            continue
        ref_times = df[ref_col_t].to_list()
        idx = df["indice"].to_list()

        other_strats = [s for s in all_strats if s != ref]
        base = [0.0] * len(idx)

        for strat in other_strats:
            strat_col_t = f"{strat}_t"
            if strat_col_t not in df.columns:
                continue
            strat_times = df[strat_col_t].to_list()
            delta_t = [s - r for s, r in zip(strat_times, ref_times)]
            color = _COLORS[strat_to_ci[strat] % len(_COLORS)]
            fig.add_trace(go.Bar(
                x=idx,
                y=delta_t,
                base=base,
                name=strat,
                legendgroup=strat,
                showlegend=(g_idx == 0),
                marker=dict(color=color),
                hovertemplate=(
                    f"<b>{strat}</b><br>idx=%{{x}}<br>"
                    "Δt=%{y:.6f}s<extra></extra>"
                ),
            ), row=r, col=c)
            base = [b + d for b, d in zip(base, delta_t)]

        fig.update_xaxes(title_text="índice", row=r, col=c)
        fig.update_yaxes(title_text="Δt acumulado (s)", row=r, col=c)
    fig.update_layout(
        title_text=f"Δt vs {ref_label}",
        height=max(420, 400 * rows), template="plotly_dark",
        margin=dict(t=150),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        barmode="relative",
    )
    return fig