"""
NSE MOMENTUM 5™ — Dashboard Page 3: Single Stock Quantitative Deep Dive
================================================================================
Provides granular quantitative factor attribution, technical chart inspection,
probability breakdowns, transparent "WHY THIS STOCK?" rationale, and interactive
fixed-rupee position sizing for any selected equity.

Components:
1. Interactive Multi-Pane Technical Candlestick Chart (OHLC, EMAs, Volume, Breakouts)
2. 8-Pillar Quantitative Momentum Score Decomposition
3. Target Move Realization Probabilities (+5%/3D, +10%/5D, +15%/5D, +20%/5D)
4. Transparent "WHY THIS STOCK?" Audit (Positive vs Cautionary Factors)
5. Interactive Position Sizer: Calculates exact shares and rupee risk
6. Quick Action: Track setup directly in Engine B Active Holdings
================================================================================
"""

from typing import Dict, Optional, Any
import pandas as pd
import streamlit as st

from dashboard.charts import render_candlestick_chart, render_probability_bar_chart
from strategy.scoring import MomentumScoringEngine
from ml.predict import TargetProbabilityEstimator
from risk.position_sizing import PositionSizer
from portfolio.holdings import HoldingsManager
from config import CONFIG


def render_stock_analysis(
    symbol: str,
    df: Optional[pd.DataFrame],
    regime_score: float = 50.0
) -> None:
    """
    Renders the in-depth quantitative factor analysis for an individual equity.
    """
    st.markdown(f"## 🔍 Stock Intelligence Deep Dive: **{symbol}**")

    if df is None or df.empty or len(df) < 20:
        st.error(f"Insufficient historical data loaded for {symbol}.")
        return

    latest_bar = df.iloc[-1]
    close_p = float(latest_bar.get("close", 0.0))
    atr = float(latest_bar.get("atr_14", close_p * 0.025))
    atr_pct = float(latest_bar.get("atr_pct", 2.5))
    rsi = float(latest_bar.get("rsi_14", 50.0))
    ret_5d = float(latest_bar.get("ret_5d", 0.0)) * 100.0
    rs_5d = float(latest_bar.get("rs_5d", 0.0)) * 100.0
    vol_r5 = float(latest_bar.get("vol_ratio_5d", 1.0))

    # Top KPI Bar
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current Price", f"INR {close_p:,.2f}")
    c2.metric("5D Return", f"{ret_5d:+.2f}%")
    c3.metric("Alpha vs NIFTY", f"{rs_5d:+.2f}%")
    c4.metric("RSI 14", f"{rsi:.1f}")
    c5.metric("Daily ATR (14)", f"INR {atr:.2f}", f"{atr_pct:.2f}%")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 1. Technical Candlestick & Volume Chart
    # --------------------------------------------------------------------------
    st.subheader("📊 Price Action, Moving Average Stack & Volume")
    fig = render_candlestick_chart(df, title=f"{symbol} Daily Technical Setup", lookback_bars=90)
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------------------------------
    # 2. 8-Pillar Quantitative Momentum Score Breakdown
    # --------------------------------------------------------------------------
    scorer = MomentumScoringEngine()
    score_res = scorer.score_record(latest_bar, regime_score=regime_score)
    bk = score_res["pillar_breakdown"]

    st.subheader(f"⚡ Quantitative Momentum Score: **{score_res['momentum_score']} / 100** ({score_res['category']})")

    row1_c1, row1_c2, row1_c3, row1_c4 = st.columns(4)
    row1_c1.metric("Price Momentum", f"{bk['price_momentum']} / 20")
    row1_c2.metric("Relative Strength", f"{bk['relative_strength']} / 15")
    row1_c3.metric("Volume Quality", f"{bk['volume_quality']} / 15")
    row1_c4.metric("Breakout Strength", f"{bk['breakout_strength']} / 15")

    row2_c1, row2_c2, row2_c3, row2_c4 = st.columns(4)
    row2_c1.metric("Trend Alignment", f"{bk['trend_alignment']} / 10")
    row2_c2.metric("Volatility Channel", f"{bk['volatility_suitability']} / 10")
    row2_c3.metric("Macro Regime", f"{bk['market_regime']} / 10")
    row2_c4.metric("Momentum Health", f"{bk['momentum_health']} / 5")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. Probability Breakdown & Transparent Evidence Rationale
    # --------------------------------------------------------------------------
    estimator = TargetProbabilityEstimator()
    probs = estimator.estimate_probabilities(symbol_df=df, current_momentum_score=score_res["momentum_score"])

    col_prob, col_why = st.columns([1, 1])

    with col_prob:
        st.subheader("🎯 Target Realization Probabilities")
        fig_prob = render_probability_bar_chart(probs)
        st.plotly_chart(fig_prob, use_container_width=True)
        st.caption(
            f"**Methodology:** {probs['methodology']} | "
            f"Expected Return: {probs.get('expected_return_pct', 0)*100:.1f}% | "
            f"Expected MAE: {probs.get('expected_adverse_excursion_pct', 0)*100:.1f}%"
        )

    with col_why:
        st.subheader("💡 Why This Stock? (Evidence Audit)")

        # Compile positive factors
        positives = []
        if ret_5d >= 3.0:
            positives.append(f"Strong 5-day velocity (+{ret_5d:.1f}%) outperforming peer universe")
        if rs_5d > 0.0:
            positives.append(f"Generating positive alpha vs NIFTY 50 (+{rs_5d:.1f}% 5D excess return)")
        if vol_r5 >= 1.3:
            positives.append(f"Institutional volume surge ({vol_r5:.1f}x 5-day average volume)")
        if latest_bar.get("breakout_20d", 0) == 1:
            positives.append("Clean close above shifted 20-day historical resistance pivot")
        if latest_bar.get("bullish_ema_stack", 0) == 1:
            positives.append("Perfect bullish moving average stack: Price > EMA5 > EMA10 > EMA20 > EMA50")
        if 55.0 <= rsi <= 75.0:
            positives.append(f"RSI 14 ({rsi:.1f}) in ideal institutional momentum acceleration zone")

        # Compile cautionary factors
        cautions = []
        if rsi > 80.0:
            cautions.append(f"Elevated RSI ({rsi:.1f}): Near-term pullback / consolidation risk")
        if atr_pct > 5.0:
            cautions.append(f"High volatility (ATR {atr_pct:.1f}%): Requires wider stop distance")
        if latest_bar.get("clv", 0.0) < 0.0:
            cautions.append("Candle closed in lower half of daily range (intraday profit booking)")

        if positives:
            for p in positives:
                st.success(f"✓ {p}")
        if cautions:
            for c in cautions:
                st.warning(f"⚠ {c}")
        if not positives and not cautions:
            st.info("Setup exhibiting neutral consolidation characteristics.")

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 4. Interactive Position Sizer & Fast Tracking
    # --------------------------------------------------------------------------
    st.subheader("📐 Precision Position Sizing Calculator")
    st.caption("Calculates exact shares based on Fixed Rupee Risk: Shares = Max Rupee Risk / Stop Distance.")

    sizer_col1, sizer_col2 = st.columns(2)

    with sizer_col1:
        capital = st.number_input(
            "Account Capital (INR)",
            min_value=50000.0,
            value=CONFIG.risk.default_capital_inr,
            step=50000.0
        )
        risk_pct = st.slider("Risk Per Trade %", min_value=0.5, max_value=3.0, value=1.0, step=0.1) / 100.0
        stop_price = st.number_input(
            "Stop Loss Price (INR)",
            min_value=1.0,
            value=round(close_p - (2.0 * atr), 2),
            step=0.5
        )

    position_sizer = PositionSizer()
    res_sizing = position_sizer.calculate_position(
        capital_inr=capital,
        entry_price=close_p,
        stop_loss_price=stop_price,
        risk_per_trade_pct=risk_pct
    )

    with sizer_col2:
        if res_sizing.is_permitted:
            st.metric("Recommended Share Count", f"{res_sizing.shares:,} shares")
            st.write(f"• **Position Value:** INR {res_sizing.position_value_inr:,.2f} ({res_sizing.portfolio_weight_pct:.1f}% of capital)")
            st.write(f"• **Total Rupee Risk:** INR {res_sizing.rupee_risk_allocated:,.2f} ({risk_pct*100:.1f}%)")
            st.write(f"• **Stop Distance:** INR {res_sizing.stop_distance_inr:.2f} ({res_sizing.stop_distance_pct*100:.1f}%)")

            if st.button("📥 Track This Position in Engine B (Active Holdings)"):
                HoldingsManager.add_position(
                    symbol=symbol,
                    entry_date=pd.Timestamp.now().to_pydatetime(),
                    entry_price=close_p,
                    quantity=res_sizing.shares,
                    stop_loss=stop_price,
                    target_price=round(close_p * 1.15, 2)
                )
                st.success(f"Successfully added {symbol} to Engine B Active Holdings!")
                st.rerun()
        else:
            st.error(f"Trade Not Permitted: {res_sizing.rejection_reason}")
