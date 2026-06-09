"""Comparación de pares: 4 subplots (correlación, tiempos, Bland-Altman, barras)."""

from pathlib import Path

from src.tui.analysis.plots.common import RESULTADOS_DIR, _hover_text


def generate_comparison_plot(df, label_a: str, label_b: str, tol: float) -> Path:
    """Genera un HTML con 4 subplots comparativos y devuelve la ruta."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    a = df["perdida_a"].to_numpy()
    b = df["perdida_b"].to_numpy()
    t_a = df["tiempo_a"].to_numpy()
    t_b = df["tiempo_b"].to_numpy()
    diff = df["diff"].to_numpy()
    idx = df["indice"].to_numpy()
    ok = df["ok"].to_numpy()
    rel = df["rel_diff"].to_numpy() * 100
    mean_a_b = (a + b) / 2
    mean_diff = float(diff.mean())
    std_diff = float(diff.std())
    mean_t_a = float(t_a.mean())
    mean_t_b = float(t_b.mean())

    colors = ["#2ecc71" if v else "#e74c3c" for v in ok]
    hover_base = _hover_text(idx, a, b, diff, rel, label_a, label_b)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Correlación de pérdidas",
            "Tiempos por índice",
            "Bland-Altman (acuerdo)",
            "Pérdidas por índice",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    # 1. Scatter correlación
    fig.add_trace(
        go.Scatter(
            x=a,
            y=b,
            mode="markers",
            marker=dict(color=colors, size=8, opacity=0.8),
            text=hover_base,
            hovertemplate="%{text}<extra></extra>",
            name="perdida",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    lo = min(a.min(), b.min()) * 0.95
    hi = max(a.max(), b.max()) * 1.05
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", color="black", width=1),
            name="y = x",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=label_a, row=1, col=1)
    fig.update_yaxes(title_text=label_b, row=1, col=1)

    # 2. Tiempos por índice
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=t_a,
            mode="lines+markers",
            name=f"{label_a} (tiempo)",
            line=dict(color="steelblue", width=2),
            marker=dict(size=5),
            hovertemplate="idx=%{x}<br>tiempo=%{y:.6f}<extra>" + label_a + "</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=t_b,
            mode="lines+markers",
            name=f"{label_b} (tiempo)",
            line=dict(color="coral", width=2),
            marker=dict(size=5),
            hovertemplate="idx=%{x}<br>tiempo=%{y:.6f}<extra>" + label_b + "</extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=mean_t_a,
        line_dash="dash",
        line_color="steelblue",
        annotation_text=f"avg {label_a}={mean_t_a:.4f}",
        row=1,
        col=2,
    )
    fig.add_hline(
        y=mean_t_b,
        line_dash="dash",
        line_color="coral",
        annotation_text=f"avg {label_b}={mean_t_b:.4f}",
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="índice", row=1, col=2)
    fig.update_yaxes(title_text="tiempo (escala log)", type="log", row=1, col=2)

    # 3. Bland-Altman
    fig.add_trace(
        go.Scatter(
            x=mean_a_b,
            y=diff,
            mode="markers",
            marker=dict(color=colors, size=8, opacity=0.8),
            text=hover_base,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    for y, label, dash in [
        (mean_diff, f"mean={mean_diff:.4f}", "solid"),
        (mean_diff + 1.96 * std_diff, "+1.96σ", "dash"),
        (mean_diff - 1.96 * std_diff, "−1.96σ", "dash"),
        (tol, f"tol={tol:.0e}", "dot"),
    ]:
        fig.add_hline(
            y=y,
            line_dash=dash,
            line_color="royalblue" if "σ" in label or "mean" in label else "orange",
            annotation_text=label,
            row=2,
            col=1,
        )
    fig.update_xaxes(title_text="Media de ambas pérdidas", row=2, col=1)
    fig.update_yaxes(title_text="|perdida_a − perdida_b|", row=2, col=1)

    # 4. Barras agrupadas
    fig.add_trace(
        go.Bar(
            x=idx,
            y=a,
            name=f"{label_a} (pérdida)",
            marker_color="steelblue",
            opacity=0.85,
            hovertemplate="idx=%{x}<br>perdida=%{y:.6f}<extra>" + label_a + "</extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=idx,
            y=b,
            name=f"{label_b} (pérdida)",
            marker_color="coral",
            opacity=0.85,
            hovertemplate="idx=%{x}<br>perdida=%{y:.6f}<extra>" + label_b + "</extra>",
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(title_text="índice", row=2, col=2)
    fig.update_yaxes(title_text="perdida", row=2, col=2)

    fig.update_layout(
        barmode="group",
        title_text=f"{label_a}  vs  {label_b}",
        title_font_size=15,
        height=800,
        template="plotly_dark",
        legend=dict(orientation="h", y=-0.05),
    )

    out_path = RESULTADOS_DIR / "compare_plot.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path
