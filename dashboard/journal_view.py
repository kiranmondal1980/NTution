"""
NSE MOMENTUM 5™ — Dashboard Page 11: Trade Journal & Audit Ledger UI
================================================================================
Comprehensive ledger of executed trades across all trading styles:
- PAPER    : Incubation paper-trading ledger
- LIVE     : Real execution audit ledger
- BACKTEST : Backtest trade records

Capabilities:
1. KPI Summary Bar (Win Rate, Net P&L, Profit Factor, Total Statutory Fees)
2. Manual Trade Entry Form
3. Filterable Relational Trade Table
4. One-Click CSV Ledger Export
5. Cumulative Realized Equity Curve Visualization
================================================================================
"""

from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from portfolio.journal import TradeJournal
from backtest.costs import NSETransactionCostCalculator


def render_journal_view() -> None:
    """
    Renders the persistent trade journal and performance ledger.
    """
    st.markdown("## 📖 Trade Journal & Audit Ledger")
    st.caption("Immutable record of executed trades across Paper, Live, and Backtest modes.")

    # --------------------------------------------------------------------------
    # 1. Manual Trade Entry Form
    # --------------------------------------------------------------------------
    with st.expander("➕ Log New Trade Entry / Exit", expanded=False):
        with st.form("manual_journal_form"):
            c1, c2, c3, c4 = st.columns(4)
            sym = c1.text_input("Symbol", "TRENT.NS")
            mode = c2.selectbox("Trade Mode", ["PAPER", "LIVE", "BACKTEST"])
            entry_p = c3.number_input("Entry Price (INR)", min_value=1.0, value=5000.0, step=1.0)
            exit_p = c4.number_input("Exit Price (INR)", min_value=1.0, value=5500.0, step=1.0)

            c5, c6, c7, c8 = st.columns(4)
            qty = c5.number_input("Quantity (Shares)", min_value=1, value=10, step=1)
            entry_dt = c6.date_input("Entry Date", value=datetime.utcnow().date())
            exit_dt = c7.date_input("Exit Date", value=datetime.utcnow().date())
            reason = c8.text_input("Exit Reason", "TARGET_10PCT_HIT")

            notes = st.text_area("Trading Notes & Observations", "Breakout above 20D pivot with volume surge.")

            submitted = st.form_submit_button("Record Trade in Ledger")
            if submitted:
                cost_calc = NSETransactionCostCalculator()
                costs = cost_calc.calculate_turnover_costs(entry_p, exit_p, qty)
                gross_pnl = (exit_p - entry_p) * qty
                net_pnl = gross_pnl - costs.total_friction
                holding_days = max(1, (exit_dt - entry_dt).days)

                TradeJournal.record_trade(
                    symbol=sym,
                    entry_date=datetime.combine(entry_dt, datetime.min.time()),
                    entry_price=entry_p,
                    exit_date=datetime.combine(exit_dt, datetime.min.time()),
                    exit_price=exit_p,
                    quantity=qty,
                    gross_pnl=gross_pnl,
                    total_costs=costs.total_friction,
                    net_pnl=net_pnl,
                    holding_days=holding_days,
                    exit_reason=reason,
                    trade_mode=mode,
                    notes=notes
                )
                st.success(f"Recorded {mode} trade for {sym} (Net P&L: INR {net_pnl:+,.2f})!")
                st.rerun()

    # --------------------------------------------------------------------------
    # 2. Mode Filter & Summary KPI Grid
    # --------------------------------------------------------------------------
    selected_mode = st.radio("Filter Ledger by Mode:", ["ALL", "PAPER", "LIVE", "BACKTEST"], horizontal=True)
    filter_arg = None if selected_mode == "ALL" else selected_mode

    kpis = TradeJournal.get_journal_kpis(trade_mode=filter_arg)

    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Journaled Trades", f"{kpis['total_trades']} trades")
    k2.metric("Win Rate", f"{kpis['win_rate_pct']:.1f}%")
    k3.metric("Cumulative Net P&L", f"INR {kpis['total_net_pnl_inr']:+,.2f}")
    k4.metric("Profit Factor", f"{kpis['profit_factor']:.2f}", f"Fees: INR {kpis['total_costs_paid_inr']:,.2f}")

    # --------------------------------------------------------------------------
    # 3. Interactive Ledger Table & CSV Export
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📜 Recorded Trade Ledger")

    df_journal = TradeJournal.get_trades_df(trade_mode=filter_arg)

    if df_journal.empty:
        st.info("No recorded trades found for the selected mode.")
        return

    st.dataframe(df_journal, use_container_width=True)

    csv_data = df_journal.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Trade Journal as CSV",
        data=csv_data,
        file_name="nse_trade_journal_ledger.csv",
        mime="text/csv"
    )

    # --------------------------------------------------------------------------
    # 4. Cumulative Realized Equity Curve
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("📈 Cumulative Realized P&L Progression")

    # Invert chronological order for cumulative sum
    df_sorted = df_journal.iloc[::-1].copy()
    df_sorted["Cumulative Net P&L (INR)"] = df_sorted["Net P&L (INR)"].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_sorted["Exit Date"],
        y=df_sorted["Cumulative Net P&L (INR)"],
        mode="lines+markers",
        line=dict(color="#00e676", width=2),
        marker=dict(size=6, color="#29b6f6"),
        name="Cumulative Net P&L"
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#90a4ae")
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        template="plotly_dark",
        yaxis_title="Realized P&L (INR)",
        xaxis_title="Exit Date"
    )
    st.plotly_chart(fig, use_container_width=True)
