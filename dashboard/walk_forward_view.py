"""
NSE MOMENTUM 5™ — Dashboard Page 7: Walk-Forward Research Results
================================================================================
Visualizes rolling out-of-sample walk-forward optimization to expose and eliminate
curve-fitting and parameter fragility across changing Indian market regimes.

Components:
1. Interactive Rolling Window Configuration (Train, Validate, Test, Step durations)
2. Overall Stability Score (0 to 100) & Walk-Forward Robustness Badge
3. Out-of-Sample Consistency % (percentage of unseen test folds generating profit)
4. Fold-by-Fold Performance Audit Table (IS vs OOS Profit Factor & Robustness Ratios)
5. Comparative Visual Bar Chart: In-Sample vs Out-of-Sample Profit Factor per Fold
================================================================================
"""

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.walk_forward import WalkForwardOptimizer
from config import CONFIG, BacktestConfig


def render_walk_forward_view(
    universe_features: Dict[str, pd.DataFrame],
    benchmark_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Renders the Walk-Forward Research & Robustness interface.
    """
    st.markdown("## 🔄 Walk-Forward Optimization & Out-of-Sample Audit")
    st.caption("Validates strategy longevity by evaluating sequential out-of-sample chronological test periods.")

    st.info(
        "📌 **OVERFITTING PREVENTION MANDATE:**\n"
        "A strategy is NOT robust simply because it showed high cumulative returns on a static backtest. "
        "It must maintain an **Out-of-Sample (OOS) Profit Factor >= 1.25** across multiple rolling temporal windows."
    )

    if not universe_features:
        st.warning("⚠️ No historical universe data loaded. Sync data in 'Settings & Data Sync'.")
        return

    # --------------------------------------------------------------------------
    # 1. Walk-Forward Window Parameter Controls
    # --------------------------------------------------------------------------
    with st.expander("⚙️ Walk-Forward Window Parameters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        train_m = c1.slider("Train Window (Months)", 12, 48, CONFIG.backtest.wf_train_months, step=6)
        val_m = c2.slider("Validation Window (Months)", 3, 12, CONFIG.backtest.wf_val_months, step=3)
        test_m = c3.slider("Out-of-Sample Test (Months)", 3, 12, CONFIG.backtest.wf_test_months, step=3)
        step_m = c4.slider("Step Size (Months)", 3, 12, CONFIG.backtest.wf_step_months, step=3)

        score_to_test = st.slider(
            "Parameter Tested: Minimum Momentum Score Trigger",
            min_value=60.0,
            max_value=85.0,
            value=CONFIG.signal.momentum,
            step=2.5
        )

    # --------------------------------------------------------------------------
    # 2. Execution Trigger
    # --------------------------------------------------------------------------
    if st.button("▶ Run Multi-Fold Walk-Forward Optimization", type="primary"):
        custom_cfg = BacktestConfig(
            wf_train_months=train_m,
            wf_val_months=val_m,
            wf_test_months=test_m,
            wf_step_months=step_m
        )

        with st.spinner("Simulating chronological rolling walk-forward folds..."):
            optimizer = WalkForwardOptimizer(cfg=custom_cfg)
            report = optimizer.run_walk_forward(
                universe_features=universe_features,
                benchmark_df=benchmark_df,
                min_momentum_score=score_to_test
            )

            if report.total_folds == 0:
                st.error(
                    "⚠️ Insufficient historical date span to construct full walk-forward folds. "
                    "Increase lookback history in Settings (at least 2 years required)."
                )
                return

            st.markdown("---")

            # ------------------------------------------------------------------
            # 3. Robustness Status Banner
            # ------------------------------------------------------------------
            if report.is_walk_forward_robust:
                st.success(
                    f"✅ **STRATEGY STATUS: WALK-FORWARD ROBUST** "
                    f"(Stability Score: {report.overall_stability_score:.1f}/100 | "
                    f"OOS Consistency: {report.consistency_pct:.1f}%)"
                )
            else:
                st.error(
                    f"⚠️ **STRATEGY STATUS: FRAGILE / NON-ROBUST** "
                    f"(Stability Score: {report.overall_stability_score:.1f}/100 | "
                    f"OOS Consistency: {report.consistency_pct:.1f}% below 70% standard)"
                )

            # ------------------------------------------------------------------
            # 4. Summary KPI Grid
            # ------------------------------------------------------------------
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Walk-Forward Folds", f"{report.total_folds} Folds")
            k2.metric("Profitable OOS Folds", f"{report.profitable_oos_folds} / {report.total_folds} ({report.consistency_pct:.1f}%)")
            k3.metric("Avg Out-of-Sample PF", f"{report.avg_oos_profit_factor:.2f}", "Benchmark >= 1.25")
            k4.metric("Worst OOS Max Drawdown", f"{report.max_oos_drawdown_pct:.1f}%", "Cap <= -25%")

            st.markdown("---")

            # ------------------------------------------------------------------
            # 5. Visual Comparison Chart: IS vs OOS Profit Factor
            # ------------------------------------------------------------------
            st.subheader("📊 In-Sample (IS) vs Out-of-Sample (OOS) Profit Factor per Fold")

            fold_labels = [f"Fold {f.fold_index}" for f in report.fold_results]
            is_pfs = [f.in_sample_summary.profit_factor for f in report.fold_results]
            oos_pfs = [f.out_of_sample_summary.profit_factor for f in report.fold_results]

            fig = go.Figure(data=[
                go.Bar(name="In-Sample PF (Training)", x=fold_labels, y=is_pfs, marker_color="#29b6f6"),
                go.Bar(name="Out-of-Sample PF (Test)", x=fold_labels, y=oos_pfs, marker_color="#00e676")
            ])
            # Add break-even reference line (PF = 1.0)
            fig.add_hline(y=1.0, line_dash="dash", line_color="#ff5252", annotation_text="Break-Even (PF=1.0)")
            fig.update_layout(
                barmode="group",
                height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                template="plotly_dark",
                yaxis_title="Profit Factor",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ------------------------------------------------------------------
            # 6. Detailed Fold-by-Fold Audit Ledger
            # ------------------------------------------------------------------
            st.subheader("📜 Granular Walk-Forward Fold Audit Ledger")

            fold_table_rows = []
            for f in report.fold_results:
                fold_table_rows.append({
                    "Fold": f.fold_index,
                    "Training Window": f"{f.window.train_start.date()} to {f.window.val_end.date()}",
                    "OOS Test Window": f"{f.window.test_start.date()} to {f.window.test_end.date()}",
                    "In-Sample PF": f.in_sample_summary.profit_factor,
                    "Out-of-Sample PF": f.out_of_sample_summary.profit_factor,
                    "Robustness Ratio": f.robustness_ratio,
                    "OOS CAGR %": f"{f.out_of_sample_summary.cagr_pct:+.1f}%",
                    "OOS Max DD %": f"{f.out_of_sample_summary.max_drawdown_pct:.1f}%",
                    "OOS Trades": f.out_of_sample_summary.total_trades,
                    "OOS Profitable": "YES" if f.is_oos_profitable else "NO"
                })

            df_folds = pd.DataFrame(fold_table_rows)
            st.dataframe(df_folds, use_container_width=True)
