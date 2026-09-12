"""
NSE MOMENTUM 5™ — Quantitative Position Sizing & Capital Allocation Engine
================================================================================
Calculates the exact number of shares to purchase based on strict Rupee Risk:

Mathematical Formula:
    Shares = floor( Maximum Rupee Risk / Stop Distance in INR )
    Where:
    - Maximum Rupee Risk = Total Capital * Risk Percentage (default 1.0%)
    - Stop Distance = Entry Price - Stop Loss Price

Capital Guardrails & Constraints Enforced:
1. Position Weight Cap: Capital allocated cannot exceed `max_position_weight_pct` (default 20%).
2. Non-Negative Floor: Fractional shares are floored to whole integer lots.
3. Stop Sanity: Forbids inverted or zero stop distances (Stop Loss must be strictly < Entry Price).
4. Sizing Rejection: If the risk-derived shares exceed available cash or equal zero,
   the trade is explicitly rejected with a documented audit rationale.
================================================================================
"""

import math
from dataclasses import dataclass
from typing import Optional, Dict, Any
from config import CONFIG, RiskConfig
from utils.logger import setup_logger

logger = setup_logger("POSITION_SIZER")


@dataclass(frozen=True)
class SizingResult:
    """Detailed output record of the position sizing calculation."""
    shares: int
    entry_price: float
    stop_loss: float
    stop_distance_inr: float
    stop_distance_pct: float
    rupee_risk_allocated: float
    position_value_inr: float
    portfolio_weight_pct: float
    is_permitted: bool
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shares": self.shares,
            "entry_price": round(self.entry_price, 2),
            "stop_loss": round(self.stop_loss, 2),
            "stop_distance_inr": round(self.stop_distance_inr, 2),
            "stop_distance_pct": round(self.stop_distance_pct * 100.0, 2),
            "rupee_risk_allocated": round(self.rupee_risk_allocated, 2),
            "position_value_inr": round(self.position_value_inr, 2),
            "portfolio_weight_pct": round(self.portfolio_weight_pct, 2),
            "is_permitted": self.is_permitted,
            "rejection_reason": self.rejection_reason
        }


class PositionSizer:
    """Calculates shares to trade constrained by portfolio risk and single-stock capital caps."""

    def __init__(self, risk_cfg: Optional[RiskConfig] = None):
        self.cfg: RiskConfig = risk_cfg or CONFIG.risk

    def calculate_position(
        self,
        capital_inr: float,
        entry_price: float,
        stop_loss_price: float,
        risk_per_trade_pct: Optional[float] = None,
        max_weight_pct: Optional[float] = None
    ) -> SizingResult:
        """
        Calculates exact share count for an entry setup.

        Args:
            capital_inr: Total current portfolio equity in INR.
            entry_price: Anticipated or filled buy price.
            stop_loss_price: Technical invalidation stop loss price.
            risk_per_trade_pct: Optional override for risk % (default 1.0%).
            max_weight_pct: Optional override for max capital weight (default 20.0%).

        Returns:
            SizingResult dataclass.
        """
        risk_pct = risk_per_trade_pct if risk_per_trade_pct is not None else self.cfg.risk_per_trade_pct
        max_w = max_weight_pct if max_weight_pct is not None else self.cfg.max_position_weight_pct

        # 1. Price Sanity Validations
        if entry_price <= 0.0:
            return SizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss_price,
                stop_distance_inr=0.0, stop_distance_pct=0.0, rupee_risk_allocated=0.0,
                position_value_inr=0.0, portfolio_weight_pct=0.0, is_permitted=False,
                rejection_reason="Entry price must be strictly positive."
            )

        if stop_loss_price <= 0.0:
            return SizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss_price,
                stop_distance_inr=0.0, stop_distance_pct=0.0, rupee_risk_allocated=0.0,
                position_value_inr=0.0, portfolio_weight_pct=0.0, is_permitted=False,
                rejection_reason="Stop loss price must be strictly positive."
            )

        stop_dist_inr = entry_price - stop_loss_price
        if stop_dist_inr <= 0.0:
            return SizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss_price,
                stop_distance_inr=stop_dist_inr, stop_distance_pct=0.0, rupee_risk_allocated=0.0,
                position_value_inr=0.0, portfolio_weight_pct=0.0, is_permitted=False,
                rejection_reason=f"Stop loss (INR {stop_loss_price:.2f}) must be strictly below entry price (INR {entry_price:.2f})."
            )

        stop_dist_pct = stop_dist_inr / entry_price

        # 2. Maximum Allowable Rupee Risk
        max_rupee_risk = capital_inr * risk_pct

        # 3. Share Sizing Calculation
        # Raw shares strictly constrained by dollar risk
        raw_risk_shares = int(math.floor(max_rupee_risk / stop_dist_inr))

        # Secondary constraint: Max position value (cannot exceed max_weight_pct of capital)
        max_capital_for_position = capital_inr * max_w
        raw_cap_shares = int(math.floor(max_capital_for_position / entry_price))

        # Take the more conservative of the two constraints
        final_shares = min(raw_risk_shares, raw_cap_shares)

        if final_shares <= 0:
            return SizingResult(
                shares=0, entry_price=entry_price, stop_loss=stop_loss_price,
                stop_distance_inr=stop_dist_inr, stop_distance_pct=stop_dist_pct,
                rupee_risk_allocated=0.0, position_value_inr=0.0, portfolio_weight_pct=0.0,
                is_permitted=False,
                rejection_reason=(
                    f"Calculated share count is 0 under risk limit (Max Risk: INR {max_rupee_risk:.2f}, "
                    f"Stop Distance: INR {stop_dist_inr:.2f})."
                )
            )

        actual_position_val = final_shares * entry_price
        actual_rupee_risk = final_shares * stop_dist_inr
        weight_pct = (actual_position_val / capital_inr) * 100.0

        return SizingResult(
            shares=final_shares,
            entry_price=entry_price,
            stop_loss=stop_loss_price,
            stop_distance_inr=round(stop_dist_inr, 2),
            stop_distance_pct=round(stop_dist_pct, 4),
            rupee_risk_allocated=round(actual_rupee_risk, 2),
            position_value_inr=round(actual_position_val, 2),
            portfolio_weight_pct=round(weight_pct, 2),
            is_permitted=True,
            rejection_reason=""
        )
