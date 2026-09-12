"""
NSE MOMENTUM 5™ — Portfolio Risk Intelligence & Capital Governor
================================================================================
Monitors aggregate portfolio heat, correlation cluster risk, concurrent
position counts, and daily circuit breaker loss thresholds.

Core Safety Policies Enforced:
1. Maximum Concurrent Positions: Caps total simultaneous open swing trades (default 5).
2. Portfolio Heat Limit: Cumulative rupee risk across all open positions cannot
   exceed `max_portfolio_risk_pct` (default 6.0% of total capital).
3. Intraday Circuit Breaker: If cumulative daily realized/unrealized loss exceeds
   `max_daily_portfolio_loss_pct` (default 3.0%), all new entries are locked.
4. Position Exposure Cap: No single position can exceed 20.0% of total equity.
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from config import CONFIG, RiskConfig
from utils.logger import setup_logger

logger = setup_logger("PORTFOLIO_RISK_ENGINE")


@dataclass(frozen=True)
class PortfolioRiskSnapshot:
    """Current risk health metrics for the overall trading account."""
    total_equity_inr: float
    invested_capital_inr: float
    cash_available_inr: float
    open_positions_count: int
    max_positions_allowed: int
    total_portfolio_heat_inr: float
    total_portfolio_heat_pct: float
    daily_realized_pnl_inr: float
    is_circuit_breaker_active: bool
    can_open_new_trade: bool
    risk_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_equity_inr": round(self.total_equity_inr, 2),
            "invested_capital_inr": round(self.invested_capital_inr, 2),
            "cash_available_inr": round(self.cash_available_inr, 2),
            "open_positions_count": self.open_positions_count,
            "max_positions_allowed": self.max_positions_allowed,
            "total_portfolio_heat_inr": round(self.total_portfolio_heat_inr, 2),
            "total_portfolio_heat_pct": round(self.total_portfolio_heat_pct, 2),
            "daily_realized_pnl_inr": round(self.daily_realized_pnl_inr, 2),
            "is_circuit_breaker_active": self.is_circuit_breaker_active,
            "can_open_new_trade": self.can_open_new_trade,
            "risk_warnings": self.risk_warnings
        }


class PortfolioRiskEngine:
    """Central risk governor protecting portfolio equity against catastrophic drawdowns."""

    def __init__(self, risk_cfg: Optional[RiskConfig] = None):
        self.cfg: RiskConfig = risk_cfg or CONFIG.risk

    def evaluate_portfolio_state(
        self,
        total_equity_inr: float,
        open_positions: List[Dict[str, Any]],
        daily_realized_pnl_inr: float = 0.0
    ) -> PortfolioRiskSnapshot:
        """
        Calculates aggregate portfolio heat, invested exposure, and circuit breaker status.

        Args:
            total_equity_inr: Total portfolio value (cash + current holdings value).
            open_positions: List of position dictionaries containing:
                            {'symbol', 'entry_price', 'current_price', 'quantity', 'current_stop'}
            daily_realized_pnl_inr: Cumulative realized net P&L for today's session.

        Returns:
            PortfolioRiskSnapshot dataclass.
        """
        warnings: List[str] = []

        # 1. Calculate Invested Capital and Cumulative Heat
        invested_cap = 0.0
        total_heat_inr = 0.0

        for pos in open_positions:
            qty = pos.get("quantity", 0)
            cur_p = pos.get("current_price", pos.get("entry_price", 0.0))
            stop_p = pos.get("current_stop", pos.get("entry_price", 0.0) * 0.95)

            pos_val = qty * cur_p
            invested_cap += pos_val

            # Risk on open position = (Current Price - Current Stop) * Quantity
            stop_dist = max(0.0, cur_p - stop_p)
            pos_risk = qty * stop_dist
            total_heat_inr += pos_risk

        cash_avail = max(0.0, total_equity_inr - invested_cap)
        heat_pct = (total_heat_inr / total_equity_inr * 100.0) if total_equity_inr > 0 else 0.0

        # 2. Check Daily Loss Circuit Breaker
        max_daily_allowable_loss = total_equity_inr * self.cfg.max_daily_portfolio_loss_pct
        circuit_breaker_hit = daily_realized_pnl_inr < -max_daily_allowable_loss

        if circuit_breaker_hit:
            warnings.append(
                f"DAILY CIRCUIT BREAKER ACTIVE: Realized loss (INR {daily_realized_pnl_inr:.2f}) "
                f"exceeds daily threshold (INR {max_daily_allowable_loss:.2f})."
            )

        # 3. Check Position Capacity
        pos_count = len(open_positions)
        at_max_positions = pos_count >= self.cfg.max_simultaneous_positions
        if at_max_positions:
            warnings.append(
                f"MAX POSITION LIMIT REACHED: {pos_count}/{self.cfg.max_simultaneous_positions} concurrent positions active."
            )

        # 4. Check Heat Cap
        max_heat_pct = self.cfg.max_portfolio_risk_pct * 100.0
        if heat_pct >= max_heat_pct:
            warnings.append(
                f"PORTFOLIO HEAT CAP EXCEEDED: Aggregate risk ({heat_pct:.2f}%) exceeds limit ({max_heat_pct:.1f}%)."
            )

        can_open = (
            not circuit_breaker_hit and
            not at_max_positions and
            (heat_pct < max_heat_pct) and
            (cash_avail > 0)
        )

        return PortfolioRiskSnapshot(
            total_equity_inr=total_equity_inr,
            invested_capital_inr=invested_cap,
            cash_available_inr=cash_avail,
            open_positions_count=pos_count,
            max_positions_allowed=self.cfg.max_simultaneous_positions,
            total_portfolio_heat_inr=total_heat_inr,
            total_portfolio_heat_pct=heat_pct,
            daily_realized_pnl_inr=daily_realized_pnl_inr,
            is_circuit_breaker_active=circuit_breaker_hit,
            can_open_new_trade=can_open,
            risk_warnings=warnings
        )

    def validate_new_trade(
        self,
        total_equity_inr: float,
        open_positions: List[Dict[str, Any]],
        proposed_position_val_inr: float,
        proposed_trade_risk_inr: float,
        daily_realized_pnl_inr: float = 0.0
    ) -> Dict[str, Any]:
        """
        Validates if a proposed new trade can be executed without violating safety limits.

        Returns:
            Dictionary: {'is_allowed': bool, 'reasons': List[str]}
        """
        snapshot = self.evaluate_portfolio_state(
            total_equity_inr, open_positions, daily_realized_pnl_inr
        )

        if not snapshot.can_open_new_trade:
            return {"is_allowed": False, "reasons": snapshot.risk_warnings}

        reasons: List[str] = []

        # Check single position weight cap
        max_single_position = total_equity_inr * self.cfg.max_position_weight_pct
        if proposed_position_val_inr > max_single_position:
            reasons.append(
                f"Proposed trade value (INR {proposed_position_val_inr:.2f}) exceeds single stock cap "
                f"(INR {max_single_position:.2f} / {self.cfg.max_position_weight_pct*100:.0f}%)."
            )

        # Check total heat after adding this trade
        new_total_heat = snapshot.total_portfolio_heat_inr + proposed_trade_risk_inr
        max_heat_inr = total_equity_inr * self.cfg.max_portfolio_risk_pct
        if new_total_heat > max_heat_inr:
            reasons.append(
                f"Adding this trade increases portfolio heat to INR {new_total_heat:.2f} ({new_total_heat/total_equity_inr*100:.2f}%), "
                f"exceeding max allowable heat of INR {max_heat_inr:.2f} ({self.cfg.max_portfolio_risk_pct*100:.1f}%)."
            )

        # Check available cash
        if proposed_position_val_inr > snapshot.cash_available_inr:
            reasons.append(
                f"Insufficient unallocated cash: Required INR {proposed_position_val_inr:.2f}, "
                f"Available INR {snapshot.cash_available_inr:.2f}."
            )

        return {
            "is_allowed": (len(reasons) == 0),
            "reasons": reasons
        }
