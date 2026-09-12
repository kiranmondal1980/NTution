"""
NSE MOMENTUM 5™ — Dashboard Page 4: Existing Holdings Dashboard
================================================================================
Comprehensive position monitoring dashboard displaying all columns required by
the quantitative architecture:

COLUMNS INCLUDED:
Symbol, Entry Date, Entry Price, Current Price, Quantity, Invested Amount,
Current Value, P&L (INR), P&L %, Highest Price, Trailing Stop, Momentum Score,
Hold Score, Trend, Relative Strength, Risk Status, Recommended Action, and Reason.
================================================================================
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd
import streamlit as st

from portfolio.holdings import HoldingsManager
from portfolio.journal import TradeJournal
from strategy.exit_engine import ExitIntelligenceEngine, ExitAction
from strategy.scoring import MomentumScoringEngine
from backtest.costs import NSETransactionCostCalculator
from config import CONFIG


def render_portfolio_view(
    universe_features: Dict[str, pd.DataFrame],
    regime_score: float = 50.0
) -> None:
    """
    Renders the active swing portfolio dashboard.
    """
    st.markdown("## 💼 Existing Holdings Dashboard")
    st.caption("Real-time position monitoring, dynamic trailing protection, and trend health Hold Scores.")

    # --------------------------------------------------------------------------
    # 1. Manual Add Position Expander
    # --------------------------------------------------------------------------
    with st.expander("➕ Register New Active Position for Tracking", expanded=False):
        with st.form("manual_add_pos_form"):
            col_a, col_b, col_c = st.columns(3)
            new_sym = col_a.text_input("NSE Symbol", "TRENT.NS")
            new_entry_p = col_b.number_input("Entry Price (INR)", min_value=1.0, value=5000.0, step=1.0)
            new_qty = col_c.number_input("Quantity (Shares)", min_value=1, value=10, step=1)

            col_d, col_e = st.columns(2)
            new_stop = col_d.number_input("Initial Stop Loss (INR)", min_value=0.5, value=round(new_entry_p * 0.95, 2), step=0.5)
            new_tgt = col_e.number_input("Upside Target (INR)", min_value=0.0, value=round(new_entry_p * 1.15, 2), step=1.0)

            submitted = st.form_submit_button("Track Position")
            if submitted:
                clean_s = new_sym.strip().upper()
                if not clean_s.endswith(".NS") and not clean_s.startswith("^"):
                    clean_s = f"{clean_s}.NS"

                HoldingsManager.add_position(
                    symbol=clean_s,
                    entry_date=datetime.utcnow(),
                    entry_price=new_entry_p,
                    quantity=new_qty,
                    stop_loss=new_stop,
                    target_price=new_tgt if new_tgt > 0 else None
                )
                st.success(f"Added {clean_s} ({new_qty} shares) to active monitoring.")
                st.rerun()

    # --------------------------------------------------------------------------
    # 2. Fetch Active Holdings & Technical Evaluations
    # --------------------------------------------------------------------------
    open_positions = HoldingsManager.get_open_positions()

    if not open_positions:
        st.info("No active holdings currently registered. Add a position above or from the Stock Analysis page.")
        return

    exit_engine = ExitIntelligenceEngine()
    scorer = MomentumScoringEngine()
    cost_calc = NSETransactionCostCalculator()

    holdings_table_rows: List[Dict[str, Any]] = []
    total_invested = 0.0
    total_current_val = 0.0
    total_unrealized_pnl = 0.0

    for pos in open_positions:
        sym = pos["symbol"].strip().upper()
        qty = pos["quantity"]
        entry_p = pos["entry_price"]
        invested = entry_p * qty
        total_invested += invested

        # Flexible lookup in universe features (matches 'ATGL.NS', 'ATGL', etc.)
        df_sym = None
        for k, v in universe_features.items():
            if k.strip().upper() == sym or k.strip().upper().replace(".NS", "") == sym.replace(".NS", ""):
                df_sym = v
                break

        if df_sym is not None and not df_sym.empty:
            latest_bar = df_sym.iloc[-1]
            cur_p = float(latest_bar.get("close", entry_p))
            trend_desc = "BULLISH STACK" if latest_bar.get("bullish_ema_stack", 0) == 1 else "ABOVE EMA20" if cur_p > float(latest_bar.get("ema_20", cur_p)) else "BELOW EMA20"
            rs_val = f"{float(latest_bar.get('rs_5d', 0))*100:+.1f}%"
            score_res = scorer.score_record(latest_bar, regime_score=regime_score)
            mom_score = score_res["momentum_score"]
        else:
            latest_bar = pd.Series()
            cur_p = entry_p
            trend_desc = "N/A"
            rs_val = "0.0%"
            mom_score = 50.0

        cur_val = cur_p * qty
        total_current_val += cur_val
        pnl_inr = cur_val - invested
        total_unrealized_pnl += pnl_inr
        pnl_pct = (pnl_inr / invested * 100.0) if invested > 0 else 0.0

        # Evaluate Exit Intelligence (Engine B)
        assessment = exit_engine.evaluate_position(
            symbol=sym,
            entry_price=entry_p,
            current_price=cur_p,
            highest_price_since_entry=max(pos["highest_price_since_entry"], cur_p),
            latest_features=latest_bar,
            user_hard_stop=pos["current_stop"],
            user_target=pos.get("target_price", 0.0)
        )

        # Ratchet database high water mark and stop
        HoldingsManager.update_position_hwm(
            position_id=pos["id"],
            latest_price=cur_p,
            active_trailing_stop=assessment.active_trailing_stop,
            hold_score=assessment.hold_score,
            exit_action=assessment.action.value
        )

        risk_stat = "NORMAL" if assessment.action in [ExitAction.HOLD, ExitAction.TRAIL] else "ELEVATED" if assessment.action == ExitAction.PARTIAL_BOOK else "CRITICAL"

        act_str = assessment.action.value
        if assessment.action == ExitAction.HOLD:
            act_badge = f"🟢 {act_str}"
        elif assessment.action == ExitAction.TRAIL:
            act_badge = f"🔵 {act_str}"
        elif assessment.action == ExitAction.PARTIAL_BOOK:
            act_badge = f"🟡 {act_str}"
        else:
            act_badge = f"🔴 {act_str}"

        holdings_table_rows.append({
            "ID": pos["id"],
            "Symbol": sym.replace(".NS", ""),
            "Entry Date": pos["entry_date"].strftime("%Y-%m-%d"),
            "Entry Price": round(entry_p, 2),
            "Current Price": round(cur_p, 2),
            "Qty": qty,
            "Invested (INR)": round(invested, 2),
            "Current Value": round(cur_val, 2),
            "P&L (INR)": round(pnl_inr, 2),
            "P&L %": round(pnl_pct, 2),
            "Highest Price": round(assessment.highest_price_since_entry, 2),
            "Trailing Stop": round(assessment.active_trailing_stop, 2),
            "Momentum Score": mom_score,
            "Hold Score": assessment.hold_score,
            "Trend": trend_desc,
            "RS vs NIFTY": rs_val,
            "Risk Status": risk_stat,
            "Recommended Action": act_badge,
            "Reason": "; ".join(assessment.primary_reasons[:2])
        })

    # --------------------------------------------------------------------------
    # 3. Portfolio Summary Metric KPIs
    # --------------------------------------------------------------------------
    pnl_total_pct = (total_unrealized_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Invested Capital", f"INR {total_invested:,.2f}")
    kpi2.metric("Current Portfolio Value", f"INR {total_current_val:,.2f}")
    kpi3.metric(
        "Total Unrealized P&L",
        f"INR {total_unrealized_pnl:,.2f}",
        f"{pnl_total_pct:+.2f}%"
    )
    kpi4.metric(
        "Active Positions Capacity",
        f"{len(open_positions)} / {CONFIG.risk.max_simultaneous_positions}",
        "Slots Open" if len(open_positions) < CONFIG.risk.max_simultaneous_positions else "Full"
    )

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 4. Formatted Table
    # --------------------------------------------------------------------------
    st.subheader(f"Monitored Holdings ({len(holdings_table_rows)} positions)")
    df_table = pd.DataFrame(holdings_table_rows)
    st.dataframe(df_table, use_container_width=True)

    # --------------------------------------------------------------------------
    # 5. Position Management & Liquidation Execution
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("⚡ Execute Position Actions")

    act_col1, act_col2, act_col3 = st.columns([1, 1, 2])
    selected_pos_id = act_col1.selectbox("Select Position ID", [r["ID"] for r in holdings_table_rows])
    action_type = act_col2.selectbox("Action", ["Full Exit / Close", "Partial Book (50%)"])

    selected_row = next((r for r in holdings_table_rows if r["ID"] == selected_pos_id), None)

    if act_col3.button("Execute Action & Record in Trade Journal"):
        if selected_row:
            cur_p = selected_row["Current Price"]
            entry_p = selected_row["Entry Price"]
            qty = selected_row["Qty"]
            sym = selected_row["Symbol"]
            entry_dt = pd.to_datetime(selected_row["Entry Date"]).to_pydatetime()
            now_dt = datetime.utcnow()
            days_held = max(1, (now_dt.date() - entry_dt.date()).days)

            if action_type == "Full Exit / Close":
                costs = cost_calc.calculate_turnover_costs(entry_p, cur_p, qty)
                gross_pnl = (cur_p - entry_p) * qty
                net_pnl = gross_pnl - costs.total_friction

                TradeJournal.record_trade(
                    symbol=sym,
                    entry_date=entry_dt,
                    entry_price=entry_p,
                    exit_date=now_dt,
                    exit_price=cur_p,
                    quantity=qty,
                    gross_pnl=gross_pnl,
                    total_costs=costs.total_friction,
                    net_pnl=net_pnl,
                    holding_days=days_held,
                    exit_reason=selected_row["Reason"],
                    trade_mode="PAPER"
                )
                HoldingsManager.close_position(selected_pos_id)
                st.success(f"Fully liquidated {sym} ({qty} shares). Trade recorded in Trade Journal.")
                st.rerun()

            elif action_type == "Partial Book (50%)":
                partial_qty = max(1, qty // 2)
                costs = cost_calc.calculate_turnover_costs(entry_p, cur_p, partial_qty)
                gross_pnl = (cur_p - entry_p) * partial_qty
                net_pnl = gross_pnl - costs.total_friction

                TradeJournal.record_trade(
                    symbol=sym,
                    entry_date=entry_dt,
                    entry_price=entry_p,
                    exit_date=now_dt,
                    exit_price=cur_p,
                    quantity=partial_qty,
                    gross_pnl=gross_pnl,
                    total_costs=costs.total_friction,
                    net_pnl=net_pnl,
                    holding_days=days_held,
                    exit_reason=f"PARTIAL BOOK (50%): {selected_row['Reason']}",
                    trade_mode="PAPER"
                )
                HoldingsManager.partial_book_position(selected_pos_id, partial_qty)
                st.success(f"Booked 50% profit on {sym} ({partial_qty} shares sold). Trade recorded in Trade Journal.")
                st.rerun()
