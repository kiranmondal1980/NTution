"""
NSE MOMENTUM 5™ — Event-Driven Chronological Portfolio Backtester
================================================================================
Simulates multi-symbol equity swing trading across historical market dates.
Strictly adheres to professional quantitative simulation standards.

ANTI-LOOKAHEAD & EXECUTION RULES:
1. Chronology Enforcement:
   - Signals calculated on Session T close.
   - Buy Orders executed strictly on Session T+1 at the Market Open + Slippage.
   - Zero same-bar fill leaks.
2. Comprehensive Trade Lifecycle:
   - Fixed Rupee Risk Position Sizing constrained by capital limits
   - Dynamic ATR trailing stop updates and staged profit protection
   - Intraday high/low tracking for target hit rate calculations
   - Time-based forced exit after 5 sessions (configurable swing horizon)
3. Friction & Slippage:
   - Exact NSE cash delivery statutory costs (Brokerage, STT, GST, Exchange, Stamp)
   - Real-world slippage deducted on both entry and exit legs
================================================================================
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from backtest.costs import NSETransactionCostCalculator
from backtest.execution import ExecutionEngine
from backtest.metrics import calculate_backtest_metrics, BacktestSummary
from strategy.scoring import MomentumScoringEngine
from strategy.exit_engine import ExitIntelligenceEngine, ExitAction
from strategy.regime import MarketRegimeEngine
from risk.position_sizing import PositionSizer
from risk.risk_engine import PortfolioRiskEngine
from config import CONFIG
from utils.logger import setup_logger

logger = setup_logger("BACKTEST_ENGINE")


class EventDrivenBacktester:
    """Institutional event-driven backtesting engine."""

    def __init__(
        self,
        initial_capital: float = CONFIG.risk.default_capital_inr,
        max_holding_sessions: int = CONFIG.backtest.default_holding_horizon_days,
        min_momentum_score: float = CONFIG.signal.momentum
    ):
        self.initial_capital: float = initial_capital
        self.max_holding_sessions: int = max_holding_sessions
        self.min_momentum_score: float = min_momentum_score

        self.cost_calc = NSETransactionCostCalculator()
        self.exec_engine = ExecutionEngine()
        self.sizer = PositionSizer()
        self.risk_engine = PortfolioRiskEngine()
        self.scorer = MomentumScoringEngine()
        self.exit_engine = ExitIntelligenceEngine()
        self.regime_engine = MarketRegimeEngine()

    def run(
        self,
        universe_features: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Executes chronological event backtesting across the universe.

        Args:
            universe_features: Dictionary of {symbol: feature_dataframe}.
            benchmark_df: Historical NIFTY 50 DataFrame.

        Returns:
            Dictionary containing:
            - 'summary': BacktestSummary dataclass
            - 'trades': List of executed trade records
            - 'equity_curve': Daily portfolio equity Series
        """
        if not universe_features:
            logger.warning("Empty universe provided to backtester.")
            return {"summary": None, "trades": [], "equity_curve": pd.Series()}

        # 1. Gather all unique chronological trading dates across universe
        all_dates = sorted(list(set.union(*[set(df.index) for df in universe_features.values()])))
        if len(all_dates) < 10:
            logger.warning("Insufficient date history to execute backtest.")
            return {"summary": None, "trades": [], "equity_curve": pd.Series()}

        logger.info(
            f"Starting backtest simulation: {all_dates[0].date()} to {all_dates[-1].date()} "
            f"({len(all_dates)} sessions, {len(universe_features)} universe tickers)..."
        )

        current_cash = self.initial_capital
        open_positions: Dict[str, Dict[str, Any]] = {}
        pending_orders: List[Dict[str, Any]] = []
        completed_trades: List[Dict[str, Any]] = []
        equity_records: Dict[pd.Timestamp, float] = {}

        # 2. Iterate chronologically day by day
        for current_date in all_dates:

            # ------------------------------------------------------------------
            # Phase A: Execute Pending Orders from Yesterday's Close at Today's Open
            # ------------------------------------------------------------------
            remaining_orders: List[Dict[str, Any]] = []
            for order in pending_orders:
                sym = order["symbol"]
                sym_df = universe_features.get(sym)

                if sym_df is not None and current_date in sym_df.index:
                    today_bar = sym_df.loc[current_date]
                    open_price = float(today_bar["open"])

                    sim_order = self.exec_engine.execute_market_open(
                        symbol=sym,
                        signal_date=order["signal_date"],
                        next_date=current_date,
                        next_open_price=open_price,
                        quantity=order["quantity"],
                        side="BUY"
                    )

                    invested_amount = sim_order.executed_price * sim_order.quantity
                    if current_cash >= invested_amount:
                        current_cash -= invested_amount
                        open_positions[sym] = {
                            "symbol": sym,
                            "entry_date": current_date,
                            "entry_price": sim_order.executed_price,
                            "quantity": sim_order.quantity,
                            "highest_price": sim_order.executed_price,
                            "lowest_price": sim_order.executed_price,
                            "hard_stop": order["hard_stop"],
                            "target_price": order["target_price"],
                            "sessions_held": 0,
                            "entry_slippage": sim_order.slippage_inr
                        }
                    else:
                        logger.debug(f"Skipping fill for {sym} on {current_date.date()}: Insufficient cash.")
                else:
                    # Stock did not trade today, carry order forward 1 session
                    remaining_orders.append(order)

            pending_orders = remaining_orders

            # ------------------------------------------------------------------
            # Phase B: Manage Active Positions & Evaluate Exits
            # ------------------------------------------------------------------
            symbols_to_close: List[str] = []

            for sym, pos in open_positions.items():
                sym_df = universe_features.get(sym)
                if sym_df is None or current_date not in sym_df.index:
                    continue

                today_bar = sym_df.loc[current_date]
                pos["sessions_held"] += 1

                b_high = float(today_bar["high"])
                b_low = float(today_bar["low"])
                b_close = float(today_bar["close"])

                pos["highest_price"] = max(pos["highest_price"], b_high)
                pos["lowest_price"] = min(pos["lowest_price"], b_low)

                # Evaluate Exit Intelligence
                exit_eval = self.exit_engine.evaluate_position(
                    symbol=sym,
                    entry_price=pos["entry_price"],
                    current_price=b_close,
                    highest_price_since_entry=pos["highest_price"],
                    latest_features=today_bar,
                    user_hard_stop=pos["hard_stop"],
                    user_target=pos["target_price"]
                )

                # Time-based expiration check (3–5 sessions holding period)
                time_expired = pos["sessions_held"] >= self.max_holding_sessions
                must_exit = exit_eval.is_terminal or time_expired

                if must_exit:
                    exit_reason = "TIME_EXPIRY_5D" if time_expired else "; ".join(exit_eval.primary_reasons)
                    exit_price = b_close

                    # Itemized turnover costs
                    costs = self.cost_calc.calculate_turnover_costs(
                        buy_price=pos["entry_price"],
                        sell_price=exit_price,
                        quantity=pos["quantity"]
                    )

                    gross_pnl = (exit_price - pos["entry_price"]) * pos["quantity"]
                    net_pnl = gross_pnl - costs.total_friction
                    net_pnl_pct = net_pnl / (pos["entry_price"] * pos["quantity"])

                    # Peak and trough excursion
                    peak_gain = (pos["highest_price"] - pos["entry_price"]) / pos["entry_price"]
                    max_loss = (pos["lowest_price"] - pos["entry_price"]) / pos["entry_price"]

                    # Research target achievement flags
                    hit_5_3 = 1 if (pos["sessions_held"] <= 3 and peak_gain >= 0.05) else 0
                    hit_10_5 = 1 if (pos["sessions_held"] <= 5 and peak_gain >= 0.10) else 0
                    hit_15_5 = 1 if (pos["sessions_held"] <= 5 and peak_gain >= 0.15) else 0
                    hit_20_5 = 1 if (pos["sessions_held"] <= 5 and peak_gain >= 0.20) else 0

                    completed_trades.append({
                        "symbol": sym,
                        "entry_date": pos["entry_date"],
                        "entry_price": pos["entry_price"],
                        "exit_date": current_date,
                        "exit_price": exit_price,
                        "quantity": pos["quantity"],
                        "gross_pnl": round(gross_pnl, 2),
                        "total_costs": costs.total_friction,
                        "slippage_inr": round(pos["entry_slippage"] + costs.slippage_cost, 2),
                        "net_pnl": round(net_pnl, 2),
                        "net_pnl_pct": round(net_pnl_pct, 4),
                        "holding_sessions": pos["sessions_held"],
                        "exit_reason": exit_reason,
                        "peak_gain_pct": round(peak_gain, 4),
                        "max_loss_pct": round(max_loss, 4),
                        "hit_5pct_3d": hit_5_3,
                        "hit_10pct_5d": hit_10_5,
                        "hit_15pct_5d": hit_15_5,
                        "hit_20pct_5d": hit_20_5
                    })

                    current_cash += (exit_price * pos["quantity"]) - costs.total_friction
                    symbols_to_close.append(sym)

            for sym in symbols_to_close:
                del open_positions[sym]

            # ------------------------------------------------------------------
            # Phase C: Mark-To-Market Total Portfolio Equity for Today
            # ------------------------------------------------------------------
            unrealized_value = 0.0
            for sym, pos in open_positions.items():
                sym_df = universe_features.get(sym)
                c_price = float(sym_df.loc[current_date]["close"]) if (sym_df is not None and current_date in sym_df.index) else pos["entry_price"]
                unrealized_value += (c_price * pos["quantity"])

            total_portfolio_equity = current_cash + unrealized_value
            equity_records[current_date] = round(total_portfolio_equity, 2)

            # ------------------------------------------------------------------
            # Phase D: Scan for New Setups at Close to Execute Tomorrow Open
            # ------------------------------------------------------------------
            # Check market regime
            regime_score = 50.0
            is_regime_permitted = True
            if benchmark_df is not None and not benchmark_df.empty:
                bench_slice = benchmark_df.loc[benchmark_df.index <= current_date]
                if len(bench_slice) >= 20:
                    r_state = self.regime_engine.evaluate_regime(bench_slice)
                    regime_score = r_state.regime_score
                    is_regime_permitted = r_state.is_long_permitted

            if is_regime_permitted:
                total_active = len(open_positions) + len(pending_orders)
                slots_available = CONFIG.risk.max_simultaneous_positions - total_active

                if slots_available > 0:
                    candidates: List[Tuple[str, float, pd.Series]] = []

                    for sym, df_sym in universe_features.items():
                        if sym in open_positions or any(p["symbol"] == sym for p in pending_orders):
                            continue
                        if current_date not in df_sym.index:
                            continue

                        bar = df_sym.loc[current_date]
                        score_res = self.scorer.score_record(bar, regime_score=regime_score)
                        score = score_res["momentum_score"]

                        if score >= self.min_momentum_score:
                            candidates.append((sym, score, bar))

                    # Rank candidates by highest momentum score
                    candidates.sort(key=lambda x: x[1], reverse=True)

                    for sym, score, bar in candidates[:slots_available]:
                        c_price = float(bar["close"])
                        atr = float(bar.get("atr_14", c_price * 0.025) or (c_price * 0.025))
                        stop_price = c_price - (2.0 * atr)

                        sizing = self.sizer.calculate_position(
                            capital_inr=total_portfolio_equity,
                            entry_price=c_price,
                            stop_loss_price=stop_price
                        )

                        if sizing.is_permitted and sizing.shares > 0:
                            # Risk governor validation
                            trade_allowed = self.risk_engine.validate_new_trade(
                                total_equity_inr=total_portfolio_equity,
                                open_positions=list(open_positions.values()),
                                proposed_position_val_inr=sizing.position_value_inr,
                                proposed_trade_risk_inr=sizing.rupee_risk_allocated
                            )

                            if trade_allowed["is_allowed"]:
                                pending_orders.append({
                                    "symbol": sym,
                                    "signal_date": current_date,
                                    "quantity": sizing.shares,
                                    "hard_stop": stop_price,
                                    "target_price": c_price * 1.15
                                })

        equity_curve = pd.Series(equity_records)
        summary = calculate_backtest_metrics(completed_trades, equity_curve, self.initial_capital)

        logger.info(
            f"Backtest completed: {summary.total_trades} trades executed. "
            f"Win Rate: {summary.win_rate_pct}%, Profit Factor: {summary.profit_factor:.2f}, "
            f"CAGR: {summary.cagr_pct:.2f}%, Max DD: {summary.max_drawdown_pct:.2f}%."
        )

        return {
            "summary": summary,
            "trades": completed_trades,
            "equity_curve": equity_curve
        }
