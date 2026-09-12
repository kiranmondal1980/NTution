"""
NSE MOMENTUM 5™ — Dashboard Page 1: Home & Market Regime Overview
================================================================================
Answers Question 1: "What is the current NSE market regime?"

Displays:
1. Macro Market Permission State: BULLISH (Green), NEUTRAL (Yellow), BEARISH (Red)
2. NIFTY 50 Technical Benchmark KPIs (Close, EMA 20, EMA 50, 5D/20D Velocity)
3. Top Systemic Candidates (Momentum, Breakout, Target Probability)
4. Existing Positions Requiring Immediate Attention (Trailing Breaches, Stops)
5. Market Breadth & Data Freshness Timestamp
================================================================================
"""

from typing import Dict, Optional, Any
import pandas as pd
import streamlit as st
from strategy.regime import MarketRegimeEngine
from strategy.scoring import MomentumScoringEngine
from portfolio.holdings import HoldingsManager
from strategy.exit_engine import ExitIntelligenceEngine, ExitAction
from dashboard.charts import render_candlestick_chart
from config import CONFIG


def render_market_overview(
    benchmark_df: Optional[pd.DataFrame],
    universe_features: Dict[str, pd.DataFrame]
) -> None:
    """
    Renders the executive institutional dashboard answering Question 1.
    """
    st.markdown("## 🌐 Home & Macro Market Regime Overview")
    st.caption("Quantitative determination of Indian equity market trend health, permission state, and urgent alerts.")

    # --------------------------------------------------------------------------
    # 1. Market Regime Evaluation
    # --------------------------------------------------------------------------
    if benchmark_df is None or benchmark_df.empty:
        st.warning(
            "⚠️ Benchmark data (NIFTY 50: `^NSEI`) not loaded. "
            "Please navigate to 'Settings & Data Sync' to initialize market history."
        )
        return

    regime_engine = MarketRegimeEngine()
    regime_state = regime_engine.evaluate_regime(benchmark_df, universe_features=universe_features)

    # Status banner with semantic color coding
    if regime_state.regime == "BULLISH":
        badge = "🟢 BULLISH (Normal Opportunity Mode — Full Risk Deployment Allowed)"
        banner_func = st.success
    elif regime_state.regime == "NEUTRAL":
        badge = "🟡 NEUTRAL (Consolidation Mode — Higher Conviction Threshold Required)"
        banner_func = st.warning
    else:
        badge = "🔴 BEARISH (Capital Preservation Mode — Aggressive Longs Gated/Throttled)"
        banner_func = st.error

    banner_func(f"**CURRENT NSE REGIME:** {badge}")

    # --------------------------------------------------------------------------
    # 2. Executive KPI Grid
    # --------------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "NIFTY 50 Close",
        f"INR {regime_state.close:,.2f}",
        f"{regime_state.nifty_5d_ret * 100:+.2f}% (5D)"
    )
    col2.metric(
        "Regime Health Score",
        f"{regime_state.regime_score:.1f} / 100",
        "Bullish >= 65" if regime_state.regime_score >= 65 else "Consolidation / Weak"
    )
    col3.metric(
        "Universe Breadth (> EMA20)",
        f"{regime_state.breadth_above_ema20_pct:.1f}%" if regime_state.breadth_above_ema20_pct is not None else "N/A",
        "Broad Participation" if (regime_state.breadth_above_ema20_pct or 0) >= 50 else "Narrow Breadth"
    )
    col4.metric(
        "Long Setups Permitted?",
        "ACTIVE (YES)" if regime_state.is_long_permitted else "SUSPENDED (NO)"
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. Urgent Holdings Attention Alerts (Engine B)
    # --------------------------------------------------------------------------
    open_positions = HoldingsManager.get_open_positions()
    urgent_alerts = []

    if open_positions and universe_features:
        exit_engine = ExitIntelligenceEngine()
        for pos in open_positions:
            sym = pos["symbol"]
            df_sym = universe_features.get(sym)
            if df_sym is not None and not df_sym.empty:
                last_bar = df_sym.iloc[-1]
                cur_price = float(last_bar["close"])
                assessment = exit_engine.evaluate_position(
                    symbol=sym,
                    entry_price=pos["entry_price"],
                    current_price=cur_price,
                    highest_price_since_entry=pos["highest_price_since_entry"],
                    latest_features=last_bar,
                    user_hard_stop=pos["current_stop"]
                )
                if assessment.action in [ExitAction.EXIT, ExitAction.EMERGENCY_EXIT, ExitAction.PARTIAL_BOOK]:
                    urgent_alerts.append((sym, assessment.action.value, assessment.primary_reasons))

    if urgent_alerts:
        st.subheader("🚨 Existing Holdings Requiring Urgent Action")
        for sym, act, reasons in urgent_alerts:
            st.error(f"**{sym}**: **{act}** — {'; '.join(reasons)}")
        st.markdown("---")

    # --------------------------------------------------------------------------
    # 4. NIFTY 50 Benchmark Technical Pane
    # --------------------------------------------------------------------------
    st.subheader("📊 Benchmark Price Action & Moving Average Stacks")
    fig_bench = render_candlestick_chart(benchmark_df, title="NIFTY 50 Benchmark (Daily)", lookback_bars=90)
    st.plotly_chart(fig_bench, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5. Top Setup Candidates Snapshot
    # --------------------------------------------------------------------------
    st.subheader("⚡ Top Momentum & Breakout Candidates (Today's Scan)")

    if universe_features:
        scorer = MomentumScoringEngine()
        candidates = []

        for sym, df_sym in universe_features.items():
            if df_sym.empty or len(df_sym) < 20:
                continue
            last_bar = df_sym.iloc[-1]
            close_p = float(last_bar["close"])
            if close_p < CONFIG.universe.min_price_inr:
                continue

            res = scorer.score_record(last_bar, regime_score=regime_state.regime_score)
            if res["momentum_score"] >= CONFIG.signal.watchlist:
                candidates.append({
                    "Symbol": sym.replace(".NS", ""),
                    "Price (INR)": round(close_p, 2),
                    "Momentum Score": res["momentum_score"],
                    "Category": res["category"],
                    "5D Ret %": f"{float(last_bar.get('ret_5d', 0))*100:+.2f}%",
                    "Alpha vs NIFTY": f"{float(last_bar.get('rs_5d', 0))*100:+.2f}%",
                    "Breakout": "20D BREAKOUT" if last_bar.get("breakout_20d", 0) == 1 else "Consolidating"
                })

        if candidates:
            cand_df = pd.DataFrame(candidates).sort_values(by="Momentum Score", ascending=False).head(5).reset_index(drop=True)
            cand_df.index += 1
            st.dataframe(cand_df, use_container_width=True)
        else:
            st.info("No candidates currently meet minimum Watchlist score thresholds.")
    else:
        st.info("Universe features not loaded. Sync historical data in Settings.")
