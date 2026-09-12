"""
NSE MOMENTUM 5™ — Dashboard Page 10: Quantitative Risk Dashboard
================================================================================
Central capital preservation and portfolio governance monitor:
1. Real-Time Portfolio Heat Monitor:
   - Tracks aggregate rupee risk across all open positions vs 6.0% cap
2. Position Capacity & Circuit Breaker Tracking:
   - Tracks concurrent open trades against max limit (5 positions)
   - Enforces daily circuit breaker loss threshold (3.0% maximum daily drawdown)
3. Precision Position Sizing Calculator:
   - Shares = floor( Maximum Rupee Risk / Stop Distance )
   - Guaranteed enforcement of single-stock capital caps (20.0% max weight)
4. Active Risk Violation Warnings & Alerts
================================================================================
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import streamlit as st

from risk.risk_engine import PortfolioRiskEngine
from risk.position_sizing import PositionSizer
from portfolio.holdings import HoldingsManager
from config import CONFIG


def render_risk_view(
    universe_features: Dict[str, pd.DataFrame]
) -> None:
    """
    Renders the quantitative portfolio risk and position governance dashboard.
    """
    st.markdown("## 🛡️ Portfolio Risk Management & Capital Governor")
    st.caption("Central risk monitor enforcing portfolio heat limits, circuit breakers, and Rupee Risk position sizing.")

    open_positions = HoldingsManager.get_open_positions()

    # --------------------------------------------------------------------------
    # 1. Total Equity Input & Current Portfolio State
    # --------------------------------------------------------------------------
    c_cap, c_day_loss = st.columns([2, 1])
    current_equity = c_cap.number_input(
        "Current Total Portfolio Equity (INR)",
        min_value=50000.0,
        value=CONFIG.risk.default_capital_inr,
        step=50000.0
    )
    daily_pnl = c_day_loss.number_input(
        "Today's Realized P&L (INR)",
        value=0.0,
        step=1000.0
    )

    # Attach current prices to open positions
    augmented_positions: List[Dict[str, Any]] = []
    for pos in open_positions:
        sym = pos["symbol"]
        df_sym = universe_features.get(sym)
        cur_p = float(df_sym.iloc[-1]["close"]) if (df_sym is not None and not df_sym.empty) else pos["entry_price"]
        augmented_positions.append({
            "symbol": sym,
            "entry_price": pos["entry_price"],
            "current_price": cur_p,
            "quantity": pos["quantity"],
            "current_stop": pos["current_stop"]
        })

    risk_engine = PortfolioRiskEngine()
    snapshot = risk_engine.evaluate_portfolio_state(
        total_equity_inr=current_equity,
        open_positions=augmented_positions,
        daily_realized_pnl_inr=daily_pnl
    )

    # --------------------------------------------------------------------------
    # 2. Risk Metrics & Circuit Breaker KPIs
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Portfolio Risk & Capacity Monitor")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Invested Capital Exposure",
        f"INR {snapshot.invested_capital_inr:,.2f}",
        f"{(snapshot.invested_capital_inr / current_equity)*100:.1f}% Allocated"
    )
    k2.metric(
        "Unallocated Cash Buffer",
        f"INR {snapshot.cash_available_inr:,.2f}",
        f"{(snapshot.cash_available_inr / current_equity)*100:.1f}% Liquid"
    )
    k3.metric(
        "Current Portfolio Heat",
        f"INR {snapshot.total_portfolio_heat_inr:,.2f}",
        f"{snapshot.total_portfolio_heat_pct:.2f}% / {CONFIG.risk.max_portfolio_risk_pct*100:.1f}% Cap"
    )
    k4.metric(
        "Concurrent Positions Active",
        f"{snapshot.open_positions_count} / {snapshot.max_positions_allowed}",
        "Slots Available" if snapshot.can_open_new_trade else "Capacity Full"
    )

    # Circuit Breaker Banner
    if snapshot.is_circuit_breaker_active:
        st.error(
            f"🚨 **CIRCUIT BREAKER TRIGGERED:** Realized loss (INR {snapshot.daily_realized_pnl_inr:,.2f}) "
            f"exceeds maximum daily loss allowance. Trading halted for the remainder of the session."
        )
    elif snapshot.risk_warnings:
        for w in snapshot.risk_warnings:
            st.warning(f"⚠️ {w}")
    else:
        st.success("✅ **PORTFOLIO RISK STATE: HEALTHY** — All risk metrics within institutional limits. New trades permitted.")

    # --------------------------------------------------------------------------
    # 3. Position-by-Position Heat Breakdown Table
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Active Holdings Risk & Stop Distance Breakdown")

    if augmented_positions:
        pos_breakdown_rows = []
        for p in augmented_positions:
            qty = p["quantity"]
            cur_p = p["current_price"]
            stop_p = p["current_stop"]
            pos_val = qty * cur_p
            pos_risk_inr = max(0.0, (cur_p - stop_p) * qty)
            pos_risk_pct = (pos_risk_inr / current_equity) * 100.0
            stop_dist_pct = ((cur_p - stop_p) / cur_p) * 100.0

            pos_breakdown_rows.append({
                "Symbol": p["symbol"].replace(".NS", ""),
                "Shares": qty,
                "Current Price (INR)": round(cur_p, 2),
                "Stop Loss (INR)": round(stop_p, 2),
                "Stop Distance %": f"{stop_dist_pct:.2f}%",
                "Position Value (INR)": round(pos_val, 2),
                "Capital Weight %": f"{(pos_val / current_equity)*100:.1f}%",
                "Rupee Risk (INR)": round(pos_risk_inr, 2),
                "Portfolio Heat %": f"{pos_risk_pct:.2f}%"
            })

        st.dataframe(pd.DataFrame(pos_breakdown_rows), use_container_width=True)
    else:
        st.info("No active open positions currently in portfolio.")

    # --------------------------------------------------------------------------
    # 4. Interactive Pre-Trade Sizing & Safety Validator
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📐 Pre-Trade Risk & Position Sizer")
    st.caption("Validates if a proposed new trade conforms to Rupee Risk and single-stock capital caps before order execution.")

    c_in1, c_in2, c_in3, c_in4 = st.columns(4)
    target_ticker = c_in1.text_input("Proposed Symbol", "RELIANCE.NS")
    proposed_entry = c_in2.number_input("Proposed Entry Price (INR)", min_value=1.0, value=2500.0, step=1.0)
    proposed_stop = c_in3.number_input("Technical Stop Loss (INR)", min_value=0.5, value=2425.0, step=0.5)
    trade_risk_pct = c_in4.slider("Risk Per Trade %", 0.5, 2.5, 1.0, step=0.1) / 100.0

    sizer = PositionSizer()
    res_calc = sizer.calculate_position(
        capital_inr=current_equity,
        entry_price=proposed_entry,
        stop_loss_price=proposed_stop,
        risk_per_trade_pct=trade_risk_pct
    )

    if res_calc.is_permitted:
        validation_check = risk_engine.validate_new_trade(
            total_equity_inr=current_equity,
            open_positions=augmented_positions,
            proposed_position_val_inr=res_calc.position_value_inr,
            proposed_trade_risk_inr=res_calc.rupee_risk_allocated,
            daily_realized_pnl_inr=daily_pnl
        )

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Recommended Share Size", f"{res_calc.shares:,} shares")
            st.write(f"• **Position Capital Value:** INR {res_calc.position_value_inr:,.2f} ({res_calc.portfolio_weight_pct:.1f}% weight)")
            st.write(f"• **Allocated Rupee Risk:** INR {res_calc.rupee_risk_allocated:,.2f} ({trade_risk_pct*100:.1f}%)")
            st.write(f"• **Stop Distance:** INR {res_calc.stop_distance_inr:.2f} ({res_calc.stop_distance_pct*100:.1f}%)")

        with res_col2:
            if validation_check["is_allowed"]:
                st.success("✅ **PRE-TRADE RISK AUDIT PASSED:** Trade conforms to portfolio heat and capacity limits.")
            else:
                st.error("❌ **PRE-TRADE RISK AUDIT FAILED:**")
                for reason in validation_check["reasons"]:
                    st.write(f"• {reason}")
    else:
        st.error(f"Sizing Calculation Invalid: {res_calc.rejection_reason}")
