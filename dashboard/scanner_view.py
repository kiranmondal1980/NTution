"""
NSE MOMENTUM 5™ — Dashboard Page 2: Quantitative Momentum Scanner
================================================================================
Answers Question 2:
"Which NSE stocks currently have the strongest statistically validated bullish short-term setup?"
and Question 3:
"What is the historical probability of achieving +5%, +10%, +15%, and +20% within 3–5 sessions?"

Features:
1. Multi-Pillar Quantitative Momentum Scoring (0 to 100)
2. Empirical & Machine-Learned Target Move Probabilities
3. Liquidity and price filters (ADTV, minimum price)
4. Comprehensive interactive ranking table with CSV export
================================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st
from strategy.scoring import MomentumScoringEngine
from ml.predict import TargetProbabilityEstimator
from config import CONFIG


def render_scanner_view(
    universe_features: Dict[str, pd.DataFrame],
    regime_score: float = 50.0
) -> None:
    """
    Renders the quantitative momentum ranking table and target probabilities.
    """
    st.markdown("## 🚀 Engine A: NSE Momentum Scanner")
    st.caption("Ranks the liquid NSE equity universe by quantitative momentum score, alpha, volume expansion, and target probabilities.")

    if not universe_features:
        st.info("No universe data loaded. Please synchronize market history in 'Settings & Data Sync'.")
        return

    # --------------------------------------------------------------------------
    # 1. Interactive Control Panel & Filters
    # --------------------------------------------------------------------------
    with st.expander("⚙️ Scanner Filters & Conviction Thresholds", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        min_score = c1.slider("Min Momentum Score", min_value=40.0, max_value=90.0, value=65.0, step=5.0)
        breakout_only = c2.checkbox("20D Breakouts Only", value=False)
        min_prob_10 = c3.slider("Min P(+10% in 5D)", min_value=0, max_value=50, value=10, step=5)
        use_ml = c4.checkbox("Use Trained ML Models (if available)", value=True)

    scorer = MomentumScoringEngine()
    estimator = TargetProbabilityEstimator()

    ranked_rows: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # 2. Iterate Across Universe Equities
    # --------------------------------------------------------------------------
    for sym, df_sym in universe_features.items():
        if df_sym.empty or len(df_sym) < 20:
            continue

        last_bar = df_sym.iloc[-1]
        close_p = float(last_bar.get("close", 0.0) or 0.0)

        # Basic liquidity check
        if close_p < CONFIG.universe.min_price_inr or close_p > CONFIG.universe.max_price_inr:
            continue

        bo_20 = int(last_bar.get("breakout_20d", 0) or 0)
        if breakout_only and bo_20 == 0:
            continue

        # Score setup
        score_res = scorer.score_record(last_bar, regime_score=regime_score)
        score = score_res["momentum_score"]
        if score < min_score:
            continue

        # Probability estimates
        probs = estimator.estimate_probabilities(
            symbol_df=df_sym,
            current_momentum_score=score,
            prefer_ml=use_ml
        )

        p10_pct = int(probs.get("prob_10pct_5d", 0.0) * 100.0)
        if p10_pct < min_prob_10:
            continue

        ret_3d = float(last_bar.get("ret_3d", 0.0) or 0.0) * 100.0
        ret_5d = float(last_bar.get("ret_5d", 0.0) or 0.0) * 100.0
        rs_5d = float(last_bar.get("rs_5d", 0.0) or 0.0) * 100.0
        vol_r5 = float(last_bar.get("vol_ratio_5d", 1.0) or 1.0)
        rsi = float(last_bar.get("rsi_14", 50.0) or 50.0)
        atr_pct = float(last_bar.get("atr_pct", 2.5) or 2.5)

        ranked_rows.append({
            "Symbol": sym.replace(".NS", ""),
            "Price (INR)": round(close_p, 2),
            "Momentum Score": score,
            "Category": score_res["category"],
            "3D Ret %": round(ret_3d, 2),
            "5D Ret %": round(ret_5d, 2),
            "Alpha vs NIFTY %": round(rs_5d, 2),
            "Vol Ratio 5D": round(vol_r5, 2),
            "RSI 14": round(rsi, 1),
            "ATR %": round(atr_pct, 2),
            "20D Breakout": "YES" if bo_20 == 1 else "NO",
            "P(+5%/3D)": f"{int(probs.get('prob_5pct_3d', 0.0)*100)}%",
            "P(+10%/5D)": f"{p10_pct}%",
            "P(+15%/5D)": f"{int(probs.get('prob_15pct_5d', 0.0)*100)}%",
            "P(+20%/5D)": f"{int(probs.get('prob_20pct_5d', 0.0)*100)}%",
            "Risk/Reward": probs.get("risk_reward_ratio", 1.2),
            "Method": probs.get("methodology", "Empirical")
        })

    # --------------------------------------------------------------------------
    # 3. Render Formatted Table & CSV Export
    # --------------------------------------------------------------------------
    if not ranked_rows:
        st.warning("No securities currently satisfy the specified screening criteria.")
        return

    table_df = pd.DataFrame(ranked_rows).sort_values(by="Momentum Score", ascending=False).reset_index(drop=True)
    table_df.index += 1  # 1-based ranking index
    table_df.index.name = "Rank"

    st.subheader(f"Ranked Momentum Opportunities ({len(table_df)} stocks)")
    st.dataframe(table_df, use_container_width=True)

    csv_data = table_df.to_csv(index=True).encode("utf-8")
    st.download_button(
        label="📥 Export Candidate Rankings as CSV",
        data=csv_data,
        file_name="nse_momentum_scanner_rankings.csv",
        mime="text/csv"
    )
