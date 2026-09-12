"""
NSE MOMENTUM 5™ — Institutional Plotly Charting Engine
================================================================================
Provides financial visual analytics tailored for quantitative equity trading:
1. Multi-Pane Technical Candlestick:
   - Price, EMA stacks (5, 10, 20, 50), Bollinger Bands, shifted breakout pivots
   - Subplot volume bars color-coded by session direction (bullish vs bearish)
2. Cumulative Equity & Drawdown Underwater Chart:
   - High water mark line + shaded underwater drawdown zone
3. Benchmark-Relative Strength (RS) Comparison:
   - Visualizes alpha divergence between target stock and NIFTY 50
4. Probability Horizon Breakdown:
   - Clean bar comparison of +5%/3D, +10%/5D, +15%/5D, and +20%/5D targets
================================================================================
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def render_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Technical Price Action & Volume",
    lookback_bars: int = 90
) -> go.Figure:
    """
    Renders an interactive multi-pane candlestick chart with moving averages and volume.
    """
    slice_df = df.iloc[-lookback_bars:] if len(df) > lookback_bars else df.copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(title, "Traded Volume"),
        row_heights=[0.75, 0.25]
    )

    # 1. Candlestick Trace
    fig.add_trace(
        go.Candlestick(
            x=slice_df.index,
            open=slice_df["open"],
            high=slice_df["high"],
            low=slice_df["low"],
            close=slice_df["close"],
            name="OHLC",
            increasing_line_color="#00e676",
            decreasing_line_color="#ff5252"
        ),
        row=1, col=1
    )

    # 2. Moving Average Overlays
    ema_configs = [
        ("ema_5", "#ffd54f", 1.0),
        ("ema_10", "#29b6f6", 1.5),
        ("ema_20", "#ab47bc", 1.8),
        ("ema_50", "#ffa726", 2.0)
    ]
    for col, color, width in ema_configs:
        if col in slice_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=slice_df.index,
                    y=slice_df[col],
                    name=col.upper(),
                    line=dict(color=color, width=width)
                ),
                row=1, col=1
            )

    # 3. Shifted Breakout Pivot Reference
    if "prior_high_20d" in slice_df.columns:
        fig.add_trace(
            go.Scatter(
                x=slice_df.index,
                y=slice_df["prior_high_20d"],
                name="20D Breakout Ref",
                line=dict(color="#90a4ae", width=1.2, dash="dash")
            ),
            row=1, col=1
        )

    # 4. Bollinger Bands (if available)
    if "bb_upper" in slice_df.columns and "bb_lower" in slice_df.columns:
        fig.add_trace(
            go.Scatter(
                x=slice_df.index,
                y=slice_df["bb_upper"],
                name="BB Upper",
                line=dict(color="rgba(120, 144, 156, 0.4)", width=1.0)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=slice_df.index,
                y=slice_df["bb_lower"],
                name="BB Lower",
                line=dict(color="rgba(120, 144, 156, 0.4)", width=1.0),
                fill="tonexty",
                fillcolor="rgba(120, 144, 156, 0.05)"
            ),
            row=1, col=1
        )

    # 5. Volume Subplot
    vol_colors = [
        "#00e676" if c >= o else "#ff5252"
        for c, o in zip(slice_df["close"], slice_df["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=slice_df.index,
            y=slice_df["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.85
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=580,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def render_equity_curve(
    equity_series: pd.Series,
    title: str = "Backtest Portfolio Equity Curve (INR)"
) -> go.Figure:
    """
    Renders the portfolio cumulative equity curve alongside underwater drawdowns.
    """
    if equity_series.empty:
        return go.Figure()

    peak = equity_series.cummax()
    drawdown_pct = ((equity_series - peak) / peak) * 100.0

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(title, "Underwater Drawdown %"),
        row_heights=[0.75, 0.25]
    )

    # 1. Equity Curve & Peak
    fig.add_trace(
        go.Scatter(
            x=equity_series.index,
            y=equity_series.values,
            mode="lines",
            line=dict(color="#00e676", width=2.2),
            name="Portfolio Equity"
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=peak.index,
            y=peak.values,
            mode="lines",
            line=dict(color="#90a4ae", width=1.0, dash="dash"),
            name="High-Water Mark"
        ),
        row=1, col=1
    )

    # 2. Underwater Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            mode="lines",
            line=dict(color="#ff5252", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(255, 82, 82, 0.25)",
            name="Drawdown %"
        ),
        row=2, col=1
    )

    fig.update_layout(
        height=480,
        margin=dict(l=10, r=10, t=35, b=10),
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def render_probability_bar_chart(prob_dict: Dict[str, float]) -> go.Figure:
    """
    Visualizes empirical or ML realization probabilities across target horizons.
    """
    categories = [
        "+5% in 3 Sessions",
        "+10% in 5 Sessions",
        "+15% in 5 Sessions",
        "+20% in 5 Sessions"
    ]
    vals = [
        prob_dict.get("prob_5pct_3d", 0.0) * 100.0,
        prob_dict.get("prob_10pct_5d", 0.0) * 100.0,
        prob_dict.get("prob_15pct_5d", 0.0) * 100.0,
        prob_dict.get("prob_20pct_5d", 0.0) * 100.0
    ]

    colors = ["#29b6f6", "#00e676", "#ffd54f", "#ff7043"]

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=vals,
            text=[f"{v:.1f}%" for v in vals],
            textposition="auto",
            marker_color=colors
        )
    ])

    fig.update_layout(
        title="Upside Target Realization Probabilities",
        yaxis=dict(title="Probability (%)", range=[0, 100]),
        height=320,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark"
    )
    return fig
