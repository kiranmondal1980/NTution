"""
NSE MOMENTUM 5™ — Dashboard Page 14: User Help & System Guide
================================================================================
Provides clear institutional guidance on how to navigate and utilize the
quantitative platform effectively.
================================================================================
"""

import streamlit as st


def render_help_view() -> None:
    """
    Renders the comprehensive user manual and guidance portal.
    """
    st.markdown("## 📖 NSE MOMENTUM 5™ — User Manual & System Guide")
    st.caption("Complete operational guide for quantitative swing traders.")

    st.markdown("---")

    # Section 1: Overview
    st.subheader("1. What is NSE MOMENTUM 5™?")
    st.write(
        "NSE MOMENTUM 5™ is a **quantitative research and decision-support platform** designed "
        "for short-term NSE equity swing traders. It focuses on identifying strong bullish momentum "
        "stocks capable of outsized upward moves over **1 to 5 trading sessions** while providing "
        "rigorous risk management and position-management intelligence."
    )

    # Section 2: The 4 Daily Questions
    st.subheader("2. The Four Daily Trading Questions")
    st.markdown(
        "Every trading day, use the platform to answer four core questions:\n"
        "* **Question 1 (Home / Market Overview):** What is the current macro NSE market regime? *(Bullish, Neutral, or Bearish)*\n"
        "* **Question 2 & 3 (Momentum Scanner):** Which stocks have the strongest statistically validated momentum, and what are their target move realization probabilities (+5%/3D, +10%/5D, +15%/5D, +20%/5D)?\n"
        "* **Question 4 (Existing Holdings / Exit Intelligence):** For stocks you already own, should you **HOLD, TRAIL, PARTIAL BOOK, or EXIT**?"
    )

    # Section 3: Step-by-Step Workflow
    st.subheader("3. Recommended Daily Workflow")
    st.markdown(
        "1. **Step 1 — Initialize Data:** Go to **Settings & Data Sync** and click **Sync Market Data Now** to download latest daily price bars.\n"
        "2. **Step 2 — Check Macro Regime:** Visit **Home / Market Overview** to verify if NIFTY 50 is bullish or bearish.\n"
        "3. **Step 3 — Run Scanner:** Open **NSE Momentum Scanner** to review ranked bullish opportunities and probability estimates.\n"
        "4. **Step 4 — Deep Dive:** Select a promising stock in **Stock Analysis Deep Dive** to inspect technical charts, factor scores, and calculate position size.\n"
        "5. **Step 5 — Monitor Holdings:** Enter your existing positions in **Existing Holdings** or **Exit Intelligence UI** to get automated trailing stops and Hold Scores."
    )

    # Section 4: FAQs
    st.subheader("4. Frequently Asked Questions (FAQ)")
    with st.expander("Q: How are target move probabilities calculated?"):
        st.write(
            "Probabilities are computed using a two-tier inference architecture: "
            "either via trained machine learning classifiers or transparent empirical conditional "
            "frequencies measuring how often similar past setups reached +5%, +10%, +15%, or +20%."
        )

    with st.expander("Q: Are returns guaranteed?"):
        st.write(
            "**No.** All figures are research targets and historical backtest results. "
            "Trading involves substantial risk of capital loss."
        )

    with st.expander("Q: What transaction costs are modeled in backtests?"):
        st.write(
            "The system deducts full statutory Indian cash delivery costs including Brokerage, "
            "STT (0.1% buy & sell), Exchange turnover charges, GST (18%), SEBI charges, Stamp Duty, "
            "and 10 bps market slippage per side."
        )
