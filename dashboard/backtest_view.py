"""
NSE MOMENTUM 5™ — Dashboard Page 6: Quantitative Backtesting Research Lab
================================================================================
Conducts event-driven chronological simulation across the liquid NSE universe.

CRITICAL QUANTITATIVE PRINCIPLES ENFORCED:
1. No Lookahead Bias: Signal generated on Close[T] -> Executed Open[T+1] + Slippage.
2. Comprehensive Cost Drag: Deducts full statutory Indian cash delivery fees:
   Brokerage + STT (0.1% Buy & Sell) + Exchange Charges + GST (18%) + Stamp Duty.
3. Transparent Attribution:
   Explicitly distinguishes [REAL DATA], [CALCULATED DATA], and [BACKTEST RESULT].
4. Multi-Condition Robustness Evaluation:
   Rejects overfit strategies. Flags robust only if positive expectancy,
   PF >= 1.30, Max DD >= -20%, and trade count >= 30.
================================================================================
"""

from typing import Dict, Optional, Any
import pandas as pd
import streamlit as st

from backtest.engine import EventDrivenBacktester
from dashboard.charts import render_equity_curve, render_probability_bar_chart
from config import CONFIG


def render_backtest_lab(
    universe_features: Dict[str, pd.DataFrame],
    benchmark_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Renders the quantitative backtest laboratory interface.
    """
    st.markdown("## 🧪 Quantitative Backtesting & Research Lab")
    st.caption("Chronologically strict event simulation with next-day market execution and full Indian statutory transaction costs.")

    st.info(
        "📌 **DATA INTEGRITY NOTICE:**\n"
        "All backtest results represent **hypothetical simulated performance** based on historical data. "
        "Execution is simulated at next-session Open with 10 bps slippage per side. "
        "Past performance is not a guarantee of future returns."
    )

    if not universe_features:
        st.warning("⚠️ No historical data available. Run 'Settings & Data Sync' before backtesting.")
        return

    # --------------------------------------------------------------------------
    # 1. Backtest Configuration Controls
    # --------------------------------------------------------------------------
    with st.expander("⚙️ Backtest Simulation Parameters & Universe Scope", expanded=True):
        col1, col2, col3 = st.columns(3)
        capital = col1.number_input(
            "Starting Capital (INR)",
            min_value=100000.0,
            value=CONFIG.risk.default_capital_inr,
            step=50000.0
        )
        holding_horizon = col2.slider("Max Holding Sessions (Time Stop)", min_value=1, max_value=10, value=5)
        score_thresh = col3.slider("Minimum Momentum Score Trigger", min_value=50.0, max_value=90.0, value=CONFIG.signal.momentum, step=2.5)

    # --------------------------------------------------------------------------
    # 2. Execution Button
    # --------------------------------------------------------------------------
    if st.button("▶ Run Full Event Backtest Simulation", type="primary"):
        with st.spinner("Executing chronological event backtest across universe..."):
            tester = EventDrivenBacktester(
                initial_capital=capital,
                max_holding_sessions=holding_horizon,
                min_momentum_score=score_thresh
            )
            results = tester.run(universe_features, benchmark_df)

            summary = results["summary"]
            trades = results["trades"]
            equity = results["equity_curve"]

            if not summary or summary.total_trades == 0:
                st.warning("No trades were generated with the selected parameter configuration.")
                return

            st.markdown("---")

            # ------------------------------------------------------------------
            # 3. Robustness Status Banner
            # ------------------------------------------------------------------
            if summary.is_robust:
                st.success("✅ **STRATEGY STATUS: ROBUST** — Strategy satisfies all institutional validation standards.")
            else:
                st.error("⚠️ **STRATEGY STATUS: NON-ROBUST / REJECTED** — Failed one or more robustness standards.")

            with st.expander("🔍 Robustness Diagnostic Audit"):
                for r in summary.robustness_reasons:
                    st.write(f"• {r}")

            # ------------------------------------------------------------------
            # 4. Primary KPI Metrics Grid
            # ------------------------------------------------------------------
            st.subheader("📊 Performance Summary [BACKTEST RESULT]")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Cumulative Return", f"{summary.cumulative_return_pct:+.2f}%")
            kpi2.metric("CAGR (Annualized)", f"{summary.cagr_pct:.2f}%")
            kpi3.metric("Profit Factor", f"{summary.profit_factor:.2f}")
            kpi4.metric("Max Drawdown", f"{summary.max_drawdown_pct:.2f}%")

            kpi5, kpi6, kpi7, kpi8 = st.columns(4)
            kpi5.metric(
                "Win Rate",
                f"{summary.win_rate_pct:.1f}%",
                f"{summary.winning_trades}W / {summary.losing_trades}L"
            )
            kpi6.metric("Sharpe Ratio", f"{summary.sharpe_ratio:.2f}")
            kpi7.metric("Sortino Ratio", f"{summary.sortino_ratio:.2f}")
            kpi8.metric("Calmar Ratio", f"{summary.calmar_ratio:.2f}")

            kpi9, kpi10, kpi11, kpi12 = st.columns(4)
            kpi9.metric("Net Expectancy / Trade", f"{summary.expectancy_pct:+.2f}%", f"INR {summary.expectancy_inr:+,.2f}")
            kpi10.metric("Avg Win / Loss Ratio", f"{summary.win_loss_ratio:.2f}")
            kpi11.metric("Avg Holding Period", f"{summary.avg_holding_sessions:.1f} sessions")
            kpi12.metric("Total Trades", f"{summary.total_trades} trades")

            # ------------------------------------------------------------------
            # 5. Research Target Hit Rates
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("🎯 Research Target Realization Hit Rates")
            st.caption("Empirical percentage of trades that reached upside benchmarks during position life:")

            t1, t2, t3, t4 = st.columns(4)
            t1.metric("+5% within 3 Sessions", f"{summary.target_hit_rate_5pct_3d:.1f}%")
            t2.metric("+10% within 5 Sessions", f"{summary.target_hit_rate_10pct_5d:.1f}%")
            t3.metric("+15% within 5 Sessions", f"{summary.target_hit_rate_15pct_5d:.1f}%")
            t4.metric("+20% within 5 Sessions", f"{summary.target_hit_rate_20pct_5d:.1f}%")

            # ------------------------------------------------------------------
            # 6. Statutory Costs & Slippage Breakdown
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("💸 Statutory Transaction Costs & Slippage Drag")
            c_cost1, c_cost2 = st.columns(2)
            c_cost1.metric("Total Statutory NSE Fees Paid", f"INR {summary.total_costs_inr:,.2f}")
            c_cost2.metric("Estimated Execution Slippage Incurred", f"INR {summary.total_slippage_inr:,.2f}")

            # ------------------------------------------------------------------
            # 7. Visual Equity Curve & Drawdown Chart
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("📈 Portfolio Equity Curve & Underwater Drawdown")
            fig_eq = render_equity_curve(equity, title=f"Portfolio Equity Curve (Start: INR {capital:,.0f})")
            st.plotly_chart(fig_eq, use_container_width=True)

            # ------------------------------------------------------------------
            # 8. Granular Executed Trade Log & CSV Export
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader(f"📜 Detailed Trade Ledger ({len(trades)} executed trades)")

            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades, use_container_width=True)

            csv_trades = df_trades.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Full Trade Log as CSV",
                data=csv_trades,
                file_name="backtest_executed_trades.csv",
                mime="text/csv"
            )
