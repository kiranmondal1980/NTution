"""
NSE MOMENTUM 5™ — Persistent Trade Journal & Audit Ledger
================================================================================
Maintains an immutable relational ledger of executed trades across all operating modes:
- BACKTEST : Historical simulation runs
- PAPER    : Live paper-trading incubation
- LIVE     : Real execution ledger

Capabilities:
1. Records complete trade execution records (Entry, Exit, P&L, Costs, Reasons, Notes)
2. Computes journal performance KPIs (Win Rate, Net P&L, Profit Factor)
3. Generates structured Pandas DataFrames for Streamlit tables and CSV exports
================================================================================
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
import pandas as pd
from database.db import get_db_session
from database.models import TradeRecord
from utils.logger import setup_logger

logger = setup_logger("TRADE_JOURNAL")


class TradeJournal:
    """Manages persistent trade records, P&L analytics, and CSV exports."""

    @staticmethod
    def record_trade(
        symbol: str,
        entry_date: datetime,
        entry_price: float,
        exit_date: datetime,
        exit_price: float,
        quantity: int,
        gross_pnl: float,
        total_costs: float,
        net_pnl: float,
        holding_days: int,
        exit_reason: str,
        trade_mode: str = "PAPER",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Inserts a completed trade record into the database.

        Args:
            symbol: Ticker symbol.
            entry_date: Inception timestamp.
            entry_price: Entry fill price.
            exit_date: Liquidation timestamp.
            exit_price: Exit fill price.
            quantity: Shares closed.
            gross_pnl: P&L before statutory friction.
            total_costs: Aggregate statutory fees and slippage.
            net_pnl: Net realized profit or loss.
            holding_days: Number of sessions held.
            exit_reason: Primary rationale for closure.
            trade_mode: "PAPER", "BACKTEST", or "LIVE".
            notes: User commentary or setup observations.

        Returns:
            Dictionary representing the inserted trade record.
        """
        invested = entry_price * quantity
        net_pnl_pct = (net_pnl / invested) if invested > 0 else 0.0

        with get_db_session() as session:
            trade = TradeRecord(
                symbol=symbol.strip().upper(),
                entry_date=entry_date,
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                quantity=quantity,
                gross_pnl=gross_pnl,
                total_costs=total_costs,
                net_pnl=net_pnl,
                net_pnl_pct=net_pnl_pct,
                holding_days=holding_days,
                exit_reason=exit_reason,
                trade_mode=trade_mode.upper(),
                notes=notes
            )
            session.add(trade)
            session.flush()

            record_dict = {
                "id": trade.id,
                "symbol": trade.symbol,
                "entry_date": trade.entry_date,
                "exit_date": trade.exit_date,
                "quantity": trade.quantity,
                "net_pnl": round(trade.net_pnl, 2),
                "net_pnl_pct": round(trade.net_pnl_pct * 100.0, 2),
                "trade_mode": trade.trade_mode,
                "exit_reason": trade.exit_reason
            }
            session.commit()

        logger.info(
            f"Journaled {trade_mode.upper()} trade: {symbol} (Net P&L: INR {net_pnl:.2f}, Return: {net_pnl_pct*100:+.2f}%)"
        )
        return record_dict

    @staticmethod
    def get_trades_df(trade_mode: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves trade records as a formatted pandas DataFrame.

        Args:
            trade_mode: Optional filter ('PAPER', 'LIVE', 'BACKTEST').
                        If None, returns all records.

        Returns:
            pd.DataFrame formatted for dashboard rendering and CSV export.
        """
        with get_db_session() as session:
            query = session.query(TradeRecord)
            if trade_mode:
                query = query.filter(TradeRecord.trade_mode == trade_mode.upper())

            records = query.order_by(TradeRecord.exit_date.desc()).all()
            if not records:
                return pd.DataFrame()

            data = [{
                "ID": r.id,
                "Mode": r.trade_mode,
                "Symbol": r.symbol,
                "Entry Date": r.entry_date.strftime("%Y-%m-%d"),
                "Entry Price (INR)": round(r.entry_price, 2),
                "Exit Date": r.exit_date.strftime("%Y-%m-%d"),
                "Exit Price (INR)": round(r.exit_price, 2),
                "Quantity": r.quantity,
                "Gross P&L (INR)": round(r.gross_pnl, 2),
                "Total Costs (INR)": round(r.total_costs, 2),
                "Net P&L (INR)": round(r.net_pnl, 2),
                "Net Return %": round(r.net_pnl_pct * 100.0, 2),
                "Holding Sessions": r.holding_days,
                "Exit Reason": r.exit_reason,
                "Notes": r.notes or ""
            } for r in records]

            return pd.DataFrame(data)

    @staticmethod
    def get_journal_kpis(trade_mode: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates aggregate key performance indicators from the trade journal.
        """
        df = TradeJournal.get_trades_df(trade_mode=trade_mode)
        if df.empty:
            return {
                "total_trades": 0,
                "win_rate_pct": 0.0,
                "total_net_pnl_inr": 0.0,
                "profit_factor": 0.0,
                "total_costs_paid_inr": 0.0
            }

        total_trades = len(df)
        wins = df[df["Net P&L (INR)"] > 0]
        losses = df[df["Net P&L (INR)"] <= 0]

        n_wins = len(wins)
        win_rate = round((n_wins / total_trades) * 100.0, 1) if total_trades > 0 else 0.0
        total_net_pnl = round(float(df["Net P&L (INR)"].sum()), 2)
        total_costs = round(float(df["Total Costs (INR)"].sum()), 2)

        gross_profit = float(wins["Net P&L (INR)"].sum()) if n_wins > 0 else 0.0
        gross_loss = float(abs(losses["Net P&L (INR)"].sum())) if len(losses) > 0 else 0.0
        if gross_loss > 0:
            profit_factor = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            profit_factor = 99.0
        else:
            profit_factor = 0.0

        return {
            "total_trades": total_trades,
            "win_rate_pct": win_rate,
            "total_net_pnl_inr": total_net_pnl,
            "profit_factor": profit_factor,
            "total_costs_paid_inr": total_costs
        }
