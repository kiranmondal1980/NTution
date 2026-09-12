"""
NSE MOMENTUM 5™ — Master Streamlit Application Entry Point
================================================================================
3–5 Day NSE India Quantitative Momentum, Probability, Risk & Exit Intelligence System.
================================================================================
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import streamlit as st

from config import CONFIG
from database.initialize import init_database
from data.downloader import MarketDataDownloader
from features import build_feature_pipeline
from strategy.regime import MarketRegimeEngine
from utils.styles import apply_custom_styling

# Dashboard View Renderers
from dashboard.market_view import render_market_overview
from dashboard.scanner_view import render_scanner_view
from dashboard.stock_view import render_stock_analysis
from dashboard.portfolio_view import render_portfolio_view
from dashboard.exit_view import render_exit_view
from dashboard.backtest_view import render_backtest_lab
from dashboard.walk_forward_view import render_walk_forward_view
from dashboard.strategy_comparison_view import render_strategy_comparison_view
from dashboard.model_probability_view import render_model_probability_view
from dashboard.risk_view import render_risk_view
from dashboard.journal_view import render_journal_view
from dashboard.settings_view import render_settings_view
from dashboard.health_view import render_health_view
from dashboard.help_view import render_help_view

# Page Configuration
st.set_page_config(
    page_title="NSE MOMENTUM 5™",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Professional UI Styling
apply_custom_styling()


@st.cache_data(ttl=1800, show_spinner=False)
def load_and_engineer_universe() -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    downloader = MarketDataDownloader()
    bench_df = downloader.load_ohlcv_from_db(CONFIG.universe.benchmark_symbol)

    universe_features: Dict[str, pd.DataFrame] = {}
    symbols = CONFIG.universe.default_symbols

    for sym in symbols:
        df = downloader.load_ohlcv_from_db(sym)
        if df.empty or len(df) < 25:
            continue
        pipeline_df = build_feature_pipeline(df, benchmark_df=bench_df)
        universe_features[sym] = pipeline_df

    return bench_df, universe_features


def main():
    init_database()

    # Sidebar Navigation
    st.sidebar.markdown("# ⚡ NSE MOMENTUM 5™")
    st.sidebar.caption("Institutional Swing Trading System")

    pages = [
        "1. Home / Market Overview",
        "2. NSE Momentum Scanner",
        "3. Stock Analysis Deep Dive",
        "4. Existing Holdings",
        "5. Exit Intelligence UI",
        "6. Backtest Lab",
        "7. Walk-Forward Results",
        "8. Strategy Comparison",
        "9. Model Probability",
        "10. Risk Dashboard",
        "11. Trade Journal",
        "12. Settings & Data Sync",
        "13. System Health Diagnostics",
        "14. User Help & Guide"
    ]

    choice = st.sidebar.radio("Navigation Menu", pages, key="main_nav_radio")

    with st.spinner("Loading market data and calculating factors..."):
        bench_df, universe_features = load_and_engineer_universe()

    regime_score = 50.0
    is_regime_permitted = True
    if bench_df is not None and not bench_df.empty and len(bench_df) >= 20:
        regime_engine = MarketRegimeEngine()
        regime_state = regime_engine.evaluate_regime(bench_df, universe_features=universe_features)
        regime_score = regime_state.regime_score
        is_regime_permitted = regime_state.is_long_permitted

    st.sidebar.markdown("---")
    st.sidebar.warning(
        "⚠️ **LEGAL DISCLAIMER:**\n"
        "Quantitative research tool. Does not guarantee profits. All trading carries risk."
    )

    # Route to Selected View (NO master title rendered above)
    if choice == "1. Home / Market Overview":
        render_market_overview(bench_df, universe_features)
    elif choice == "2. NSE Momentum Scanner":
        render_scanner_view(universe_features, regime_score=regime_score)
    elif choice == "3. Stock Analysis Deep Dive":
        available_syms = sorted(list(universe_features.keys())) if universe_features else []
        if not available_syms:
            st.info("No universe data loaded. Sync data in Settings.")
        else:
            c1, _ = st.columns([1, 2])
            selected_sym = c1.selectbox("Select Target Stock:", available_syms)
            render_stock_analysis(selected_sym, universe_features[selected_sym], regime_score=regime_score)
    elif choice == "4. Existing Holdings":
        render_portfolio_view(universe_features, regime_score=regime_score)
    elif choice == "5. Exit Intelligence UI":
        render_exit_view(universe_features, regime_permitted=is_regime_permitted)
    elif choice == "6. Backtest Lab":
        render_backtest_lab(universe_features, benchmark_df=bench_df)
    elif choice == "7. Walk-Forward Results":
        render_walk_forward_view(universe_features, benchmark_df=bench_df)
    elif choice == "8. Strategy Comparison":
        render_strategy_comparison_view(universe_features, benchmark_df=benchmark_df)
    elif choice == "9. Model Probability":
        render_model_probability_view(universe_features)
    elif choice == "10. Risk Dashboard":
        render_risk_view(universe_features)
    elif choice == "11. Trade Journal":
        render_journal_view()
    elif choice == "12. Settings & Data Sync":
        render_settings_view()
    elif choice == "13. System Health Diagnostics":
        render_health_view()
    elif choice == "14. User Help & Guide":
        render_help_view()


if __name__ == "__main__":
    main()
