"""
NSE MOMENTUM 5™
============================================================
Master Streamlit Application Entry Point

3–5 Day NSE India Quantitative Momentum,
Probability, Risk & Exit Intelligence System.

This module is responsible for:
    • Application configuration
    • Global UI shell
    • Navigation
    • Market-data loading
    • Feature-engineering orchestration
    • Market-regime calculation
    • Dashboard routing

Trading logic remains inside the dedicated strategy/risk modules.
============================================================
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from config import CONFIG
from database.initialize import init_database
from data.downloader import MarketDataDownloader
from features import build_feature_pipeline
from strategy.regime import MarketRegimeEngine
from utils.styles import apply_custom_styling


# ============================================================
# DASHBOARD VIEW IMPORTS
# ============================================================

from dashboard.market_view import render_market_overview
from dashboard.scanner_view import render_scanner_view
from dashboard.stock_view import render_stock_analysis
from dashboard.portfolio_view import render_portfolio_view
from dashboard.exit_view import render_exit_view
from dashboard.backtest_view import render_backtest_lab
from dashboard.walk_forward_view import render_walk_forward_view
from dashboard.strategy_comparison_view import (
    render_strategy_comparison_view,
)
from dashboard.model_probability_view import (
    render_model_probability_view,
)
from dashboard.risk_view import render_risk_view
from dashboard.journal_view import render_journal_view
from dashboard.settings_view import render_settings_view
from dashboard.health_view import render_health_view
from dashboard.help_view import render_help_view


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="NSE MOMENTUM 5™",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": (
            "NSE MOMENTUM 5™ — Quantitative NSE India "
            "Momentum & Exit Intelligence System."
        ),
    },
)


# ============================================================
# GLOBAL STYLING
# ============================================================

apply_custom_styling()


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state() -> None:
    """Initialize application-level Streamlit session state."""

    defaults = {
        "main_nav": "Market Overview",
        "last_loaded_at": None,
        "data_load_error": None,
        "database_initialized": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# NAVIGATION DEFINITIONS
# ============================================================

NAVIGATION = {
    "TRADING": [
        ("Market Overview", "📊"),
        ("Momentum Scanner", "🚀"),
        ("Stock Analysis", "🔎"),
        ("Existing Holdings", "💼"),
        ("Exit Intelligence", "🛡️"),
    ],
    "RESEARCH": [
        ("Backtest Lab", "🧪"),
        ("Walk-Forward Results", "📈"),
        ("Strategy Comparison", "⚖️"),
        ("Model Probability", "🎯"),
    ],
    "RISK & RECORDS": [
        ("Risk Dashboard", "⚠️"),
        ("Trade Journal", "📓"),
    ],
    "SYSTEM": [
        ("Settings & Data Sync", "⚙️"),
        ("System Health", "🩺"),
        ("User Guide", "📘"),
    ],
}


PAGE_TO_RENDERER = {
    "Market Overview": "market",
    "Momentum Scanner": "scanner",
    "Stock Analysis": "stock",
    "Existing Holdings": "portfolio",
    "Exit Intelligence": "exit",
    "Backtest Lab": "backtest",
    "Walk-Forward Results": "walk_forward",
    "Strategy Comparison": "strategy_comparison",
    "Model Probability": "model_probability",
    "Risk Dashboard": "risk",
    "Trade Journal": "journal",
    "Settings & Data Sync": "settings",
    "System Health": "health",
    "User Guide": "help",
}


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_text(value: object, fallback: str = "—") -> str:
    """Return a display-safe string."""
    if value is None:
        return fallback

    text = str(value).strip()

    if not text:
        return fallback

    return text


def get_benchmark_symbol() -> str:
    """Return configured benchmark symbol safely."""

    try:
        return safe_text(
            CONFIG.universe.benchmark_symbol,
            "NIFTY 50",
        )
    except AttributeError:
        return "NIFTY 50"


def get_universe_symbols() -> list[str]:
    """Return configured universe symbols safely."""

    try:
        symbols = CONFIG.universe.default_symbols

        if symbols is None:
            return []

        return list(symbols)

    except AttributeError:
        return []


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(
    ttl=1800,
    show_spinner=False,
)
def load_and_engineer_universe(
) -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Load benchmark and equity data from the configured database
    and calculate the feature pipeline.

    Returns
    -------
    Tuple[
        Optional[pd.DataFrame],
        Dict[str, pd.DataFrame]
    ]
        Benchmark dataframe and symbol -> engineered dataframe.
    """

    downloader = MarketDataDownloader()

    benchmark_symbol = get_benchmark_symbol()

    bench_df = downloader.load_ohlcv_from_db(
        benchmark_symbol
    )

    universe_features: Dict[str, pd.DataFrame] = {}

    symbols = get_universe_symbols()

    for symbol in symbols:

        try:
            df = downloader.load_ohlcv_from_db(symbol)

            if df is None or df.empty:
                continue

            if len(df) < 25:
                continue

            pipeline_df = build_feature_pipeline(
                df,
                benchmark_df=bench_df,
            )

            if (
                pipeline_df is not None
                and not pipeline_df.empty
            ):
                universe_features[symbol] = pipeline_df

        except Exception:
            # One broken symbol must not stop the entire universe.
            continue

    return bench_df, universe_features


# ============================================================
# DATABASE
# ============================================================

def ensure_database() -> bool:
    """Initialize database once per Streamlit session."""

    if st.session_state.database_initialized:
        return True

    try:
        init_database()

        st.session_state.database_initialized = True

        return True

    except Exception as exc:
        st.session_state.data_load_error = (
            f"Database initialization failed: {exc}"
        )

        return False


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(
    bench_df: Optional[pd.DataFrame],
    universe_features: Dict[str, pd.DataFrame],
) -> Tuple[float, bool, str]:
    """
    Calculate market regime information.

    Returns
    -------
    regime_score
    is_long_permitted
    regime_label
    """

    default_score = 50.0
    default_permitted = True
    default_label = "NEUTRAL"

    if bench_df is None or bench_df.empty:
        return (
            default_score,
            default_permitted,
            default_label,
        )

    if len(bench_df) < 20:
        return (
            default_score,
            default_permitted,
            default_label,
        )

    try:
        regime_engine = MarketRegimeEngine()

        regime_state = regime_engine.evaluate_regime(
            bench_df,
            universe_features=universe_features,
        )

        score = float(
            getattr(
                regime_state,
                "regime_score",
                default_score,
            )
        )

        permitted = bool(
            getattr(
                regime_state,
                "is_long_permitted",
                default_permitted,
            )
        )

        raw_label = getattr(
            regime_state,
            "regime",
            None,
        )

        if raw_label is None:
            raw_label = getattr(
                regime_state,
                "regime_label",
                None,
            )

        label = safe_text(
            raw_label,
            default_label,
        ).upper()

        return score, permitted, label

    except Exception:
        return (
            default_score,
            default_permitted,
            default_label,
        )


# ============================================================
# HEADER
# ============================================================

def render_application_header(
    regime_score: float,
    regime_label: str,
    data_available: bool,
    symbol_count: int,
) -> None:
    """Render the compact institutional trading-terminal header."""

    data_status = (
        "LIVE DATA"
        if data_available
        else "DATA OFFLINE"
    )

    status_class = (
        "status-live"
        if data_available
        else "status-warning"
    )

    st.markdown(
        f"""
        <div class="nm5-topbar">
            <div class="nm5-brand">
                <div class="nm5-logo">⚡</div>
                <div>
                    <div class="nm5-brand-name">
                        NSE MOMENTUM 5<span class="nm5-tm">™</span>
                    </div>
                    <div class="nm5-brand-subtitle">
                        SHORT-TERM QUANTITATIVE TRADING TERMINAL
                    </div>
                </div>
            </div>
            <div class="nm5-header-status">
                <div class="nm5-status-item">
                    <span class="nm5-status-label">MARKET</span>
                    <span class="nm5-status-value">{regime_label}</span>
                </div>
                <div class="nm5-status-item">
                    <span class="nm5-status-label">REGIME SCORE</span>
                    <span class="nm5-status-value">{regime_score:.0f}/100</span>
                </div>
                <div class="nm5-status-item">
                    <span class="nm5-status-label">UNIVERSE</span>
                    <span class="nm5-status-value">{symbol_count:,}</span>
                </div>
                <div class="nm5-status-item">
                    <span class="nm5-status-label">DATA</span>
                    <span class="{status_class}">● {data_status}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> str:
    """Render the structured trading-terminal navigation."""

    with st.sidebar:

        st.markdown(
            """
            <div class="nm5-sidebar-brand">
                <div class="nm5-sidebar-mark">⚡</div>
                <div>
                    <div class="nm5-sidebar-title">
                        MOMENTUM 5
                    </div>
                    <div class="nm5-sidebar-subtitle">
                        NSE INDIA
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="nm5-sidebar-section-title">'
            "TRADING"
            "</div>",
            unsafe_allow_html=True,
        )

        selected_page = None

        for section, items in NAVIGATION.items():

            if section != "TRADING":
                st.markdown(
                    f'<div class="nm5-sidebar-section-title">'
                    f"{section}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            for page_name, icon in items:

                page_key = f"nav_{page_name}"

                is_active = (
                    st.session_state.main_nav == page_name
                )

                button_label = (
                    f"{icon}  {page_name}"
                )

                if st.button(
                    button_label,
                    key=page_key,
                    use_container_width=True,
                    type=(
                        "primary"
                        if is_active
                        else "secondary"
                    ),
                ):
                    selected_page = page_name

        if selected_page:
            st.session_state.main_nav = selected_page

        st.markdown("---")

        # Quick actions
        st.markdown(
            '<div class="nm5-sidebar-section-title">'
            "QUICK ACTIONS"
            "</div>",
            unsafe_allow_html=True,
        )

        q1, q2 = st.columns(2)

        with q1:
            if st.button(
                "🚀\nSCAN",
                key="quick_scan",
                use_container_width=True,
            ):
                st.session_state.main_nav = "Momentum Scanner"

        with q2:
            if st.button(
                "🛡️\nEXIT",
                key="quick_exit",
                use_container_width=True,
            ):
                st.session_state.main_nav = "Exit Intelligence"

        # --- THIS IS THE FIX (added unsafe_allow_html=True) ---
        st.markdown(
            """
            <div class="nm5-sidebar-footer">
                <div class="nm5-sidebar-risk">
                    ⚠️ RESEARCH & DECISION SUPPORT
                </div>

                <div class="nm5-sidebar-disclaimer">
                    Historical analysis does not guarantee
                    future returns. Trading involves risk.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state.main_nav

# ============================================================
# DATA STATUS BAR
# ============================================================

def render_data_status_bar(
    bench_df: Optional[pd.DataFrame],
    universe_features: Dict[str, pd.DataFrame],
) -> None:
    """Display data health and freshness information."""

    has_benchmark = (
        bench_df is not None
        and not bench_df.empty
    )

    has_universe = bool(universe_features)

    data_ok = has_benchmark and has_universe

    if data_ok:

        latest_date = None

        try:
            if "Date" in bench_df.columns:
                latest_date = bench_df["Date"].max()

            elif isinstance(bench_df.index, pd.DatetimeIndex):
                latest_date = bench_df.index.max()

        except Exception:
            latest_date = None

        latest_display = safe_text(
            latest_date,
            "Available",
        )

        st.markdown(
            f"""
            <div class="nm5-data-bar nm5-data-ok">
                <span>●</span>
                <strong>DATA READY</strong>
                <span class="nm5-data-separator">|</span>
                Benchmark: {get_benchmark_symbol()}
                <span class="nm5-data-separator">|</span>
                Symbols: {len(universe_features):,}
                <span class="nm5-data-separator">|</span>
                Latest: {latest_display}
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="nm5-data-bar nm5-data-warning">
                <span>●</span>
                <strong>DATA NOT READY</strong>
                <span class="nm5-data-separator">|</span>
                Go to Settings & Data Sync to load market data.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# PAGE TITLE
# ============================================================

def render_page_heading(
    page_name: str,
    description: str,
) -> None:
    """Render consistent page heading."""

    st.markdown(
        f"""
        <div class="nm5-page-heading">
            <div>
                <div class="nm5-page-kicker">
                    NSE MOMENTUM 5™
                </div>
                <h1>{page_name}</h1>
                <div class="nm5-page-description">
                    {description}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,  # <--- CRITICAL FIX
    )


PAGE_DESCRIPTIONS = {
    "Market Overview":
        "Market regime, breadth, momentum and trading environment.",
    "Momentum Scanner":
        "Rank NSE stocks by short-term quantitative momentum.",
    "Stock Analysis":
        "Deep-dive into price action, factors, probability and risk.",
    "Existing Holdings":
        "Monitor open positions, P&L and trend health.",
    "Exit Intelligence":
        "Determine whether an existing position should hold, trail or exit.",
    "Backtest Lab":
        "Evaluate strategy performance using historical data.",
    "Walk-Forward Results":
        "Test strategy stability across sequential out-of-sample periods.",
    "Strategy Comparison":
        "Compare alternative systematic strategy families.",
    "Model Probability":
        "Evaluate historical and machine-learning target probabilities.",
    "Risk Dashboard":
        "Monitor portfolio exposure, position risk and drawdown.",
    "Trade Journal":
        "Record, review and analyse trading decisions.",
    "Settings & Data Sync":
        "Configure the system and manage market-data synchronization.",
    "System Health":
        "Check database, data, dependencies and system integrity.",
    "User Guide":
        "Understand the system, signals, risks and workflow.",
}


# ============================================================
# STOCK ANALYSIS ROUTER
# ============================================================

def render_stock_page(
    universe_features: Dict[str, pd.DataFrame],
    regime_score: float,
) -> None:
    """Render stock selection before the existing stock renderer."""

    available_symbols = sorted(
        universe_features.keys()
    )

    if not available_symbols:

        st.warning(
            "No engineered stock data is currently available."
        )

        st.info(
            "Open Settings & Data Sync to load historical market data."
        )

        return

    selected_symbol = st.selectbox(
        "SELECT STOCK",
        available_symbols,
        key="stock_analysis_symbol",
        label_visibility="collapsed",
    )

    if selected_symbol:

        st.markdown(
            f"""
            <div class="nm5-selected-stock">
                <span class="nm5-selected-label">
                    SELECTED SECURITY
                </span>
                <span class="nm5-selected-symbol">
                    {selected_symbol}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_stock_analysis(
            selected_symbol,
            universe_features[selected_symbol],
            regime_score=regime_score,
        )


# ============================================================
# PAGE ROUTER
# ============================================================

def route_page(
    page: str,
    bench_df: Optional[pd.DataFrame],
    universe_features: Dict[str, pd.DataFrame],
    regime_score: float,
    is_regime_permitted: bool,
) -> None:
    """Route the selected navigation item."""

    if page == "Market Overview":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_market_overview(
            bench_df,
            universe_features,
        )

    elif page == "Momentum Scanner":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_scanner_view(
            universe_features,
            regime_score=regime_score,
        )

    elif page == "Stock Analysis":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_stock_page(
            universe_features,
            regime_score,
        )

    elif page == "Existing Holdings":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_portfolio_view(
            universe_features,
            regime_score=regime_score,
        )

    elif page == "Exit Intelligence":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_exit_view(
            universe_features,
            regime_permitted=is_regime_permitted,
        )

    elif page == "Backtest Lab":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_backtest_lab(
            universe_features,
            benchmark_df=bench_df,
        )

    elif page == "Walk-Forward Results":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_walk_forward_view(
            universe_features,
            benchmark_df=bench_df,
        )

    elif page == "Strategy Comparison":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_strategy_comparison_view(
            universe_features,
            benchmark_df=bench_df,
        )

    elif page == "Model Probability":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_model_probability_view(
            universe_features,
        )

    elif page == "Risk Dashboard":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_risk_view(
            universe_features,
        )

    elif page == "Trade Journal":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_journal_view()

    elif page == "Settings & Data Sync":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_settings_view()

    elif page == "System Health":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_health_view()

    elif page == "User Guide":

        render_page_heading(
            page,
            PAGE_DESCRIPTIONS[page],
        )

        render_help_view()

    else:

        st.error(
            f"Unknown application page: {page}"
        )


# ============================================================
# APPLICATION FOOTER
# ============================================================

def render_footer() -> None:
    """Render application footer."""

    current_year = datetime.now().year

    st.markdown(
        f"""
        <div class="nm5-footer">
            <div>
                <strong>NSE MOMENTUM 5™</strong>
                &nbsp;|&nbsp;
                Quantitative Research & Decision Support
            </div>
            <div>
                © {current_year}
                &nbsp;•&nbsp;
                Historical performance is not indicative
                of future results.
            </div>
        </div>
        """,
        unsafe_allow_html=True,  # <--- CRITICAL FIX
    )

# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:
    """Main Streamlit application entry point."""

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------

    database_ok = ensure_database()

    if not database_ok:

        st.error(
            "The application database could not be initialized."
        )

        st.info(
            "Open System Health for diagnostics."
        )

    # --------------------------------------------------------
    # Navigation
    # --------------------------------------------------------

    selected_page = render_sidebar()

    # --------------------------------------------------------
    # Load Data
    # --------------------------------------------------------

    bench_df: Optional[pd.DataFrame] = None
    universe_features: Dict[str, pd.DataFrame] = {}

    data_load_error = None

    try:

        with st.spinner(
            "Loading market data and calculating quantitative factors..."
        ):

            (
                bench_df,
                universe_features,
            ) = load_and_engineer_universe()

        st.session_state.last_loaded_at = datetime.now()
        st.session_state.data_load_error = None

    except Exception as exc:

        data_load_error = str(exc)

        st.session_state.data_load_error = data_load_error

    # --------------------------------------------------------
    # Market Regime
    # --------------------------------------------------------

    (
        regime_score,
        is_regime_permitted,
        regime_label,
    ) = calculate_market_regime(
        bench_df,
        universe_features,
    )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    render_application_header(
        regime_score=regime_score,
        regime_label=regime_label,
        data_available=bool(universe_features),
        symbol_count=len(universe_features),
    )

    # --------------------------------------------------------
    # Data status
    # --------------------------------------------------------

    render_data_status_bar(
        bench_df,
        universe_features,
    )

    # --------------------------------------------------------
    # Load error
    # --------------------------------------------------------

    if data_load_error:

        st.error(
            "Market-data loading encountered an error."
        )

        with st.expander(
            "Technical details",
            expanded=False,
        ):
            st.code(
                data_load_error,
                language="text",
            )

    # --------------------------------------------------------
    # Main content
    # --------------------------------------------------------

    route_page(
        page=selected_page,
        bench_df=bench_df,
        universe_features=universe_features,
        regime_score=regime_score,
        is_regime_permitted=is_regime_permitted,
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    render_footer()


# ============================================================
# APPLICATION ENTRY
# ============================================================

if __name__ == "__main__":
    main()
