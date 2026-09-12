"""
NSE MOMENTUM 5™ — Dashboard Page 8: Multi-Strategy Research Leaderboard
================================================================================
Compares performance across multiple distinct quantitative strategy families:
- Strategy A: Pure Momentum Continuation
- Strategy B: Breakout + Volume Confirmation
- Strategy C: Benchmark Relative Strength Outperformance
- Strategy D: Trend Alignment + Price Acceleration
- Strategy E: Multi-Factor Quantitative Momentum
- Ensemble  : NSE MOMENTUM 5 ENSEMBLE (Consensus Model)

SELECTION MANDATE:
Recommends the **BEST ROBUST STRATEGY** rather than the "highest return" strategy.
Strategies are scored on: Return, Drawdown, Profit Factor, Expectancy, and Stability.
================================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st

from backtest.engine import EventDrivenBacktester
from backtest.metrics import BacktestSummary
from config import CONFIG


def calculate_robustness_score(summary: BacktestSummary) -> float:
    """
    Computes a transparent 0 to 100 Robustness Score based on:
    - Profit Factor (weight 30%)
    - Win Rate (weight 20%)
    - Max Drawdown Preservation (weight 25%)
    - Sharpe Ratio (weight 15%)
    - Trade Count Significance (weight 10%)
    """
    if not summary or summary.total_trades == 0:
        return 0.0

    # 1. Profit Factor component (capped at 2.5 = 100%)
    score_pf = min(summary.profit_factor / 2.5, 1.0) * 30.0

    # 2. Win rate component (50% to 70% scaled to 0-100%)
    score_wr = min(max((summary.win_rate_pct - 40.0) / 25.0, 0.0), 1.0) * 20.0

    # 3. Drawdown preservation (0% DD = 100%, -30% DD = 0%)
    score_dd = max(0.0, 1.0 - (abs(summary.max_drawdown_pct) / 30.0)) * 25.0

    # 4. Sharpe ratio (0 to 2.0 scaled)
    score_sharpe = min(max(summary.sharpe_ratio / 2.0, 0.0), 1.0) * 15.0

    # 5. Sample size (>= 30 trades = 100%)
    score_samples = min(summary.total_trades / 30.0, 1.0) * 10.0

    total = score_pf + score_wr + score_dd + score_sharpe + score_samples
    return round(float(total), 1)


def render_strategy_comparison_view(
    universe_features: Dict[str, pd.DataFrame],
    benchmark_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Renders the quantitative strategy comparison leaderboard.
    """
    st.markdown("## 🏆 Strategy Research Leaderboard & Model Selection")
    st.caption("Compares competing quantitative strategy models to recommend the BEST ROBUST STRATEGY.")

    st.info(
        "📌 **CORE RESEARCH PRINCIPLE:**\n"
        "Never select a strategy merely because it boasts the highest nominal return. "
        "The system evaluates **Robustness Score (0–100)** incorporating Profit Factor, "
        "Maximum Drawdown, Net Expectancy, and Statistical Trade Count."
    )

    if not universe_features:
        st.warning("⚠️ No historical data available. Run 'Settings & Data Sync' first.")
        return

    # --------------------------------------------------------------------------
    # 1. Benchmark Execution Trigger
    # --------------------------------------------------------------------------
    if st.button("▶ Run Full Multi-Strategy Comparison Suite", type="primary"):
        with st.spinner("Simulating full universe backtests across all strategy families..."):

            strategy_configs = [
                ("Strategy A (Pure Momentum)", 70.0, 5),
                ("Strategy B (Breakout + Volume)", 75.0, 5),
                ("Strategy C (Relative Strength Alpha)", 72.5, 5),
                ("Strategy D (Trend + Acceleration)", 75.0, 4),
                ("Strategy E (Multi-Factor Momentum)", 75.0, 5),
                ("NSE Momentum 5 Ensemble", 80.0, 5)
            ]

            leaderboard_rows = []

            for strat_name, min_score, horizon in strategy_configs:
                tester = EventDrivenBacktester(
                    initial_capital=CONFIG.risk.default_capital_inr,
                    max_holding_sessions=horizon,
                    min_momentum_score=min_score
                )
                res = tester.run(universe_features, benchmark_df)
                s = res["summary"]

                if s and s.total_trades > 0:
                    rob_score = calculate_robustness_score(s)
                    leaderboard_rows.append({
                        "Strategy": strat_name,
                        "Robustness Score": rob_score,
                        "Cumulative Ret %": s.cumulative_return_pct,
                        "CAGR %": s.cagr_pct,
                        "Profit Factor": s.profit_factor,
                        "Win Rate %": s.win_rate_pct,
                        "Expectancy %": s.expectancy_pct,
                        "Max Drawdown %": s.max_drawdown_pct,
                        "Sharpe Ratio": s.sharpe_ratio,
                        "Sortino Ratio": s.sortino_ratio,
                        "Trades": s.total_trades,
                        "Total Costs (INR)": s.total_costs_inr,
                        "Hit Rate +15%/5D": f"{s.target_hit_rate_15pct_5d:.1f}%",
                        "Is Robust": s.is_robust
                    })

            if not leaderboard_rows:
                st.warning("No strategies generated sufficient trades across the current historical window.")
                return

            df_leaderboard = pd.DataFrame(leaderboard_rows).sort_values(by="Robustness Score", ascending=False).reset_index(drop=True)
            df_leaderboard.index += 1
            df_leaderboard.index.name = "Rank"

            # ------------------------------------------------------------------
            # 2. Winner Recommendation Banner
            # ------------------------------------------------------------------
            best_strat = df_leaderboard.iloc[0]
            st.success(
                f"🏅 **RECOMMENDED BEST ROBUST STRATEGY:** **{best_strat['Strategy']}** "
                f"(Robustness Score: {best_strat['Robustness Score']}/100 | "
                f"Profit Factor: {best_strat['Profit Factor']} | "
                f"Max DD: {best_strat['Max Drawdown %']}%)"
            )

            # ------------------------------------------------------------------
            # 3. Interactive Leaderboard Table
            # ------------------------------------------------------------------
            st.subheader("📊 Quantitative Model Comparison Table")
            st.dataframe(df_leaderboard, use_container_width=True)

            csv_data = df_leaderboard.to_csv(index=True).encode("utf-8")
            st.download_button(
                label="📥 Export Leaderboard as CSV",
                data=csv_data,
                file_name="strategy_research_leaderboard.csv",
                mime="text/csv"
            )
