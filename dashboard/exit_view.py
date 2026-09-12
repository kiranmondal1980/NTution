"""
NSE MOMENTUM 5™ — Dashboard Page 5: Engine B Exit Intelligence UI
================================================================================
Dedicated standalone position evaluator answering:
"I already own this stock. Should I HOLD, TRAIL, PARTIAL BOOK, or EXIT?"

Capabilities:
1. Custom Position Inspection: Evaluate any existing position by symbol & entry price
2. Displays:
   - Current Unrealized P&L
   - Hold Score (0 to 100) representing trend structure health
   - Dynamic ATR Trailing Stop & Staged Profit Protection Tier
   - Definitive Single Recommendation Badge:
     🟢 HOLD | 🔵 TRAIL | 🟡 PARTIAL BOOK | 🔴 EXIT | 🚨 EMERGENCY EXIT
   - Machine-Generated Evidence Audit Rationale
3. Interactive "What-If" Scenario Simulator: Tests trailing boundaries dynamically
================================================================================
"""

from typing import Dict, Optional, Any
from datetime import datetime
import pandas as pd
import streamlit as st

from strategy.exit_engine import ExitIntelligenceEngine, ExitAction
from portfolio.holdings import HoldingsManager
from config import CONFIG


def render_exit_view(
    universe_features: Dict[str, pd.DataFrame],
    regime_permitted: bool = True
) -> None:
    """
    Renders the dedicated Exit Intelligence interface.
    """
    st.markdown("## 🛡️ Engine B: Exit Intelligence & Trailing Engine")
    st.caption("Evidence-based holding analysis, dynamic ATR stops, and staged profit protection.")

    exit_engine = ExitIntelligenceEngine()

    # --------------------------------------------------------------------------
    # 1. Position Input Selector
    # --------------------------------------------------------------------------
    open_positions = HoldingsManager.get_open_positions()

    mode = st.radio(
        "Select Evaluation Mode:",
        ["Inspect Tracked Position from Database", "Ad-Hoc / Custom Position Calculator"],
        horizontal=True
    )

    if mode == "Inspect Tracked Position from Database":
        if not open_positions:
            st.info("No active holdings found in database. Switch to 'Ad-Hoc / Custom Position Calculator' below.")
            return

        pos_map = {f"{p['symbol']} (Entry: INR {p['entry_price']:.2f}, Qty: {p['quantity']})": p for p in open_positions}
        selected_label = st.selectbox("Select Existing Holding:", list(pos_map.keys()))
        selected_pos = pos_map[selected_label]

        symbol = selected_pos["symbol"]
        entry_p = selected_pos["entry_price"]
        qty = selected_pos["quantity"]
        entry_dt = selected_pos["entry_date"]
        user_stop = selected_pos["current_stop"]
        user_tgt = selected_pos.get("target_price", 0.0) or 0.0
        hwm_price = selected_pos.get("highest_price_since_entry", entry_p)

    else:
        available_symbols = sorted(list(universe_features.keys())) if universe_features else ["TRENT.NS"]
        c1, c2, c3 = st.columns(3)
        symbol = c1.selectbox("NSE Symbol", available_symbols)
        entry_p = c2.number_input("My Entry Price (INR)", min_value=1.0, value=5000.0, step=1.0)
        qty = c3.number_input("Shares Owned", min_value=1, value=10, step=1)

        c4, c5, c6 = st.columns(3)
        user_stop = c4.number_input("My Initial Stop (INR, Optional)", min_value=0.0, value=round(entry_p * 0.95, 2))
        user_tgt = c5.number_input("My Target (INR, Optional)", min_value=0.0, value=round(entry_p * 1.15, 2))
        hwm_price = c6.number_input("Highest Price Reached Since Entry", min_value=entry_p, value=entry_p)
        entry_dt = datetime.utcnow()

    # --------------------------------------------------------------------------
    # 2. Fetch Latest Technical Indicators
    # --------------------------------------------------------------------------
    df_sym = universe_features.get(symbol)
    if df_sym is None or df_sym.empty:
        st.warning(f"No market data found for {symbol}. Run Data Sync in Settings.")
        return

    latest_bar = df_sym.iloc[-1]
    cur_price = float(latest_bar.get("close", entry_p))
    peak_price = max(hwm_price, cur_price)

    assessment = exit_engine.evaluate_position(
        symbol=symbol,
        entry_price=entry_p,
        current_price=cur_price,
        highest_price_since_entry=peak_price,
        latest_features=latest_bar,
        user_hard_stop=user_stop,
        user_target=user_tgt,
        regime_permitted=regime_permitted
    )

    unrealized_pnl = (cur_price - entry_p) * qty
    pnl_pct = assessment.unrealized_pnl_pct * 100.0

    st.markdown("---")

    # --------------------------------------------------------------------------
    # 3. Action Recommendation Header
    # --------------------------------------------------------------------------
    act = assessment.action
    if act == ExitAction.HOLD:
        rec_color = st.success
        badge = "🟢 HOLD POSITION"
    elif act == ExitAction.TRAIL:
        rec_color = st.info
        badge = "🔵 TRAIL STOP (TIGHTEN PROTECTION)"
    elif act == ExitAction.PARTIAL_BOOK:
        rec_color = st.warning
        badge = "🟡 PARTIAL BOOK (50% PROFIT HARVEST)"
    elif act == ExitAction.EMERGENCY_EXIT:
        rec_color = st.error
        badge = "🚨 EMERGENCY EXIT (HARD STOP BREACH)"
    else:
        rec_color = st.error
        badge = "🔴 FULL EXIT (TREND COLLAPSED)"

    rec_color(f"### PRIMARY ACTION: {badge}")

    # --------------------------------------------------------------------------
    # 4. Status KPIs
    # --------------------------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Current Unrealized P&L",
        f"INR {unrealized_pnl:+,.2f}",
        f"{pnl_pct:+.2f}%"
    )
    k2.metric(
        "Trend Hold Score",
        f"{assessment.hold_score} / 100",
        assessment.hold_category
    )
    k3.metric(
        "Dynamic Trailing Stop",
        f"INR {assessment.active_trailing_stop:.2f}",
        f"{((assessment.active_trailing_stop - cur_price)/cur_price)*100:.1f}% below"
    )
    k4.metric(
        "Peak High Reached",
        f"INR {peak_price:.2f}",
        f"{((peak_price - entry_p)/entry_p)*100:+.1f}% from entry"
    )

    # --------------------------------------------------------------------------
    # 5. Profit Protection Stage & Primary Evidence
    # --------------------------------------------------------------------------
    st.markdown("---")
    c_tier, c_why = st.columns([1, 2])

    with c_tier:
        st.subheader("🛡️ Profit Protection Status")
        st.info(f"**Active Tier:**\n\n{assessment.profit_tier}")
        st.write(f"• **Effective Hard Stop:** INR {assessment.hard_stop:.2f}")
        st.write(f"• **Active Trailing Stop:** INR {assessment.active_trailing_stop:.2f}")

    with c_why:
        st.subheader("💡 Calculated Decision Evidence")
        for reason in assessment.primary_reasons:
            if "violated" in reason.lower() or "breached" in reason.lower() or "below" in reason.lower():
                st.error(f"• {reason}")
            elif "holding" in reason.lower() or "robust" in reason.lower():
                st.success(f"• {reason}")
            else:
                st.write(f"• {reason}")

    # --------------------------------------------------------------------------
    # 6. Interactive "What-If" Scenario Simulator
    # --------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("🧪 'What-If' Price Scenario Simulation")
    st.caption("Slide the hypothetical market price to observe how trailing stops and Hold Scores dynamically react.")

    sim_price = st.slider(
        "Simulated Tomorrow Price (INR)",
        min_value=round(cur_price * 0.85, 2),
        max_value=round(cur_price * 1.25, 2),
        value=round(cur_price, 2),
        step=0.5
    )

    sim_assessment = exit_engine.evaluate_position(
        symbol=symbol,
        entry_price=entry_p,
        current_price=sim_price,
        highest_price_since_entry=max(peak_price, sim_price),
        latest_features=latest_bar,
        user_hard_stop=user_stop,
        user_target=user_tgt,
        regime_permitted=regime_permitted
    )

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Simulated Action", sim_assessment.action.value)
    sc2.metric("Simulated Hold Score", f"{sim_assessment.hold_score} / 100")
    sc3.metric("Simulated Trailing Stop", f"INR {sim_assessment.active_trailing_stop:.2f}")
