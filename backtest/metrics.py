"""
NSE MOMENTUM 5™ — Quantitative Backtesting Performance Metrics Engine
================================================================================
Calculates statistical and financial metrics from simulated trade ledgers and
daily equity curves.

METRICS EVALUATED:
1. Core Trading Statistics:
   - Total Trades, Winning Trades, Losing Trades, Win Rate %
   - Average Win %, Average Loss %, Win/Loss Ratio
   - Profit Factor: Gross Profit / Gross Loss
   - Net Expectancy (in INR and percentage per trade)
2. Risk-Adjusted Return Metrics:
   - Cumulative Return % & Compounded Annual Growth Rate (CAGR %)
   - Maximum Drawdown % & Peak-to-Trough Drawdown Duration
   - Annualized Sharpe Ratio (vs 6.50% Indian Sovereign Yield benchmark)
   - Annualized Sortino Ratio (Downside deviation risk)
   - Calmar Ratio: CAGR / Max Drawdown
3. Operational & Holding Dynamics:
   - Average & Median Holding Period (trading sessions)
   - Largest Single Win %, Largest Single Loss %
   - Maximum Consecutive Wins & Maximum Consecutive Losses
   - Total Transaction Costs & Slippage Impact in INR
4. Research Target Realization Hit Rates:
   - Hit Rate: +5% within 3 sessions
   - Hit Rate: +10% within 5 sessions
   - Hit Rate: +15% within 5 sessions
   - Hit Rate: +20% within 5 sessions
5. Maximum Favorable / Adverse Excursion (MFE / MAE)
6. Institutional Robustness Standards:
   - Evaluates multi-condition robustness criteria to reject curve-fit strategies.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from config import CONFIG, BacktestConfig


@dataclass(frozen=True)
class BacktestSummary:
    """Comprehensive performance report for a backtest simulation."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    expectancy_inr: float
    expectancy_pct: float
    cumulative_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_trade_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    win_loss_ratio: float
    largest_win_pct: float
    largest_loss_pct: float
    avg_holding_sessions: float
    median_holding_sessions: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    total_costs_inr: float
    total_slippage_inr: float
    avg_mfe_pct: float                   # Max Favorable Excursion
    avg_mae_pct: float                   # Max Adverse Excursion
    target_hit_rate_5pct_3d: float       # Research target hit rates
    target_hit_rate_10pct_5d: float
    target_hit_rate_15pct_5d: float
    target_hit_rate_20pct_5d: float
    is_robust: bool                      # Multi-condition robustness standard
    robustness_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate_pct": self.win_rate_pct,
            "profit_factor": self.profit_factor,
            "expectancy_inr": self.expectancy_inr,
            "expectancy_pct": self.expectancy_pct,
            "cumulative_return_pct": self.cumulative_return_pct,
            "cagr_pct": self.cagr_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "avg_trade_pnl_pct": self.avg_trade_pnl_pct,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "win_loss_ratio": self.win_loss_ratio,
            "largest_win_pct": self.largest_win_pct,
            "largest_loss_pct": self.largest_loss_pct,
            "avg_holding_sessions": self.avg_holding_sessions,
            "median_holding_sessions": self.median_holding_sessions,
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
            "total_costs_inr": self.total_costs_inr,
            "total_slippage_inr": self.total_slippage_inr,
            "avg_mfe_pct": self.avg_mfe_pct,
            "avg_mae_pct": self.avg_mae_pct,
            "target_hit_rate_5pct_3d": self.target_hit_rate_5pct_3d,
            "target_hit_rate_10pct_5d": self.target_hit_rate_10pct_5d,
            "target_hit_rate_15pct_5d": self.target_hit_rate_15pct_5d,
            "target_hit_rate_20pct_5d": self.target_hit_rate_20pct_5d,
            "is_robust": self.is_robust,
            "robustness_reasons": self.robustness_reasons
        }


def calculate_backtest_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: pd.Series,
    initial_capital: float = CONFIG.risk.default_capital_inr,
    cfg: Optional[BacktestConfig] = None
) -> BacktestSummary:
    """
    Computes institutional performance metrics from executed trade records and daily equity curve.

    Args:
        trades: List of trade dictionaries.
        equity_curve: Daily mark-to-market total portfolio equity Series.
        initial_capital: Starting capital in INR.
        cfg: Configuration parameters for robustness standards.

    Returns:
        BacktestSummary dataclass.
    """
    bt_cfg: BacktestConfig = cfg or CONFIG.backtest

    # Handle zero trades condition safely
    if not trades:
        return BacktestSummary(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate_pct=0.0,
            profit_factor=0.0, expectancy_inr=0.0, expectancy_pct=0.0,
            cumulative_return_pct=0.0, cagr_pct=0.0, max_drawdown_pct=0.0,
            max_drawdown_duration_days=0, sharpe_ratio=0.0, sortino_ratio=0.0,
            calmar_ratio=0.0, avg_trade_pnl_pct=0.0, avg_win_pct=0.0, avg_loss_pct=0.0,
            win_loss_ratio=0.0, largest_win_pct=0.0, largest_loss_pct=0.0,
            avg_holding_sessions=0.0, median_holding_sessions=0.0, max_consecutive_wins=0,
            max_consecutive_losses=0, total_costs_inr=0.0, total_slippage_inr=0.0,
            avg_mfe_pct=0.0, avg_mae_pct=0.0, target_hit_rate_5pct_3d=0.0,
            target_hit_rate_10pct_5d=0.0, target_hit_rate_15pct_5d=0.0,
            target_hit_rate_20pct_5d=0.0, is_robust=False,
            robustness_reasons=["No trades executed in simulation."]
        )

    df = pd.DataFrame(trades)
    total_trades = len(df)

    # 1. Trade Categorization
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    n_wins = len(wins)
    n_losses = len(losses)
    win_rate = (n_wins / total_trades) * 100.0 if total_trades > 0 else 0.0

    # 2. Profit Factor & Win/Loss Ratio
    gross_profit = float(wins["net_pnl"].sum()) if n_wins > 0 else 0.0
    gross_loss = float(abs(losses["net_pnl"].sum())) if n_losses > 0 else 0.0
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 99.0
    else:
        profit_factor = 0.0

    avg_win = float(wins["net_pnl_pct"].mean()) if n_wins > 0 else 0.0
    avg_loss = float(losses["net_pnl_pct"].mean()) if n_losses > 0 else 0.0
    win_loss_ratio = abs(avg_win / avg_loss) if abs(avg_loss) > 0 else 0.0

    # 3. Mathematical Expectancy
    expectancy_pct = (win_rate / 100.0 * avg_win) + ((1.0 - win_rate / 100.0) * avg_loss)
    expectancy_inr = float(df["net_pnl"].mean())

    # 4. Returns & Drawdown from Daily Equity Curve
    max_dd = 0.0
    cagr = 0.0
    sharpe = 0.0
    sortino = 0.0
    calmar = 0.0
    cum_ret = 0.0
    max_dd_duration = 0

    if not equity_curve.empty and len(equity_curve) > 1:
        # Cumulative return
        ending_eq = float(equity_curve.iloc[-1])
        cum_ret = ((ending_eq / initial_capital) - 1.0) * 100.0

        # Drawdown calculation
        peak = equity_curve.cummax()
        dd_series = (equity_curve - peak) / peak
        max_dd = float(dd_series.min()) * 100.0  # negative %

        # Max Drawdown Duration (days in drawdown)
        is_in_dd = dd_series < 0
        dd_streaks = is_in_dd.groupby((~is_in_dd).cumsum()).cumsum()
        max_dd_duration = int(dd_streaks.max()) if len(dd_streaks) > 0 else 0

        # CAGR
        days_span = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = max(days_span / 365.25, 0.1)
        if ending_eq > 0:
            cagr = (((ending_eq / initial_capital) ** (1.0 / years)) - 1.0) * 100.0

        # Sharpe & Sortino (vs 6.50% Indian G-Sec risk-free rate)
        daily_returns = equity_curve.pct_change().dropna()
        rf_daily = bt_cfg.annual_risk_free_rate / 252.0
        excess_daily = daily_returns - rf_daily

        std_dev = excess_daily.std()
        if std_dev > 0:
            sharpe = float((excess_daily.mean() / std_dev) * np.sqrt(252))

        downside = excess_daily[excess_daily < 0]
        downside_std = downside.std()
        if downside_std > 0:
            sortino = float((excess_daily.mean() / downside_std) * np.sqrt(252))

        # Calmar Ratio: CAGR / abs(Max DD)
        if abs(max_dd) > 0:
            calmar = abs(cagr / max_dd)

    # 5. Holding Dynamics
    holding_sessions = df.get("holding_sessions", pd.Series([1]*total_trades))
    avg_holding = float(holding_sessions.mean())
    med_holding = float(holding_sessions.median())

    # 6. Consecutive Wins / Losses
    pnl_signs = np.where(df["net_pnl"] > 0, 1, -1)
    max_w_streak = 0
    max_l_streak = 0
    cur_w = 0
    cur_l = 0
    for sign in pnl_signs:
        if sign == 1:
            cur_w += 1
            cur_l = 0
            max_w_streak = max(max_w_streak, cur_w)
        else:
            cur_l += 1
            cur_w = 0
            max_l_streak = max(max_l_streak, cur_l)

    # 7. Costs & Slippage
    total_costs = float(df["total_costs"].sum()) if "total_costs" in df.columns else 0.0
    total_slippage = float(df["slippage_inr"].sum()) if "slippage_inr" in df.columns else 0.0

    # 8. Research Target Realization Hit Rates
    hit_5_3 = float(df.get("hit_5pct_3d", pd.Series([0]*total_trades)).mean()) * 100.0
    hit_10_5 = float(df.get("hit_10pct_5d", pd.Series([0]*total_trades)).mean()) * 100.0
    hit_15_5 = float(df.get("hit_15pct_5d", pd.Series([0]*total_trades)).mean()) * 100.0
    hit_20_5 = float(df.get("hit_20pct_5d", pd.Series([0]*total_trades)).mean()) * 100.0

    # 9. MFE / MAE
    avg_mfe = float(df.get("peak_gain_pct", pd.Series([0.0]*total_trades)).mean()) * 100.0
    avg_mae = float(df.get("max_loss_pct", pd.Series([0.0]*total_trades)).mean()) * 100.0

    # 10. Multi-Condition Robustness Evaluation
    robustness_notes: List[str] = []
    is_robust = True

    if total_trades < bt_cfg.min_closed_trades:
        is_robust = False
        robustness_notes.append(f"Insufficient trade count ({total_trades} < {bt_cfg.min_closed_trades}).")

    if profit_factor < bt_cfg.min_profit_factor:
        is_robust = False
        robustness_notes.append(f"Profit Factor ({profit_factor:.2f}) < {bt_cfg.min_profit_factor:.2f}.")

    if expectancy_pct < bt_cfg.min_out_of_sample_expectancy_pct:
        is_robust = False
        robustness_notes.append(f"Net Expectancy ({expectancy_pct*100:.2f}%) < {bt_cfg.min_out_of_sample_expectancy_pct*100:.2f}%.")

    if max_dd < bt_cfg.max_allowable_drawdown_pct:
        is_robust = False
        robustness_notes.append(f"Max Drawdown ({max_dd:.1f}%) breaches limit ({bt_cfg.max_allowable_drawdown_pct:.1f}%).")

    if sharpe < bt_cfg.min_sharpe_ratio:
        is_robust = False
        robustness_notes.append(f"Sharpe Ratio ({sharpe:.2f}) < {bt_cfg.min_sharpe_ratio:.2f}.")

    if is_robust:
        robustness_notes.append("PASSED: Strategy satisfies institutional robustness standards.")

    return BacktestSummary(
        total_trades=total_trades,
        winning_trades=n_wins,
        losing_trades=n_losses,
        win_rate_pct=round(win_rate, 2),
        profit_factor=round(profit_factor, 2),
        expectancy_inr=round(expectancy_inr, 2),
        expectancy_pct=round(expectancy_pct * 100.0, 2),
        cumulative_return_pct=round(cum_ret, 2),
        cagr_pct=round(cagr, 2),
        max_drawdown_pct=round(max_dd, 2),
        max_drawdown_duration_days=max_dd_duration,
        sharpe_ratio=round(sharpe, 2),
        sortino_ratio=round(sortino, 2),
        calmar_ratio=round(calmar, 2),
        avg_trade_pnl_pct=round(float(df["net_pnl_pct"].mean()) * 100.0, 2),
        avg_win_pct=round(avg_win * 100.0, 2),
        avg_loss_pct=round(avg_loss * 100.0, 2),
        win_loss_ratio=round(win_loss_ratio, 2),
        largest_win_pct=round(float(df["net_pnl_pct"].max()) * 100.0, 2),
        largest_loss_pct=round(float(df["net_pnl_pct"].min()) * 100.0, 2),
        avg_holding_sessions=round(avg_holding, 1),
        median_holding_sessions=round(med_holding, 1),
        max_consecutive_wins=max_w_streak,
        max_consecutive_losses=max_l_streak,
        total_costs_inr=round(total_costs, 2),
        total_slippage_inr=round(total_slippage, 2),
        avg_mfe_pct=round(avg_mfe, 2),
        avg_mae_pct=round(avg_mae, 2),
        target_hit_rate_5pct_3d=round(hit_5_3, 2),
        target_hit_rate_10pct_5d=round(hit_10_5, 2),
        target_hit_rate_15pct_5d=round(hit_15_5, 2),
        target_hit_rate_20pct_5d=round(hit_20_5, 2),
        is_robust=is_robust,
        robustness_reasons=robustness_notes
    )
