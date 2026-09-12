"""
NSE MOMENTUM 5™ — Institutional Indian Equity Transaction Cost Engine
================================================================================
Calculates exact real-world transaction friction for NSE cash delivery trades.
All rates are parameterizable via `config.py`.

STATUTORY & BROKERAGE FEE BREAKDOWN:
1. Brokerage:
   - Configured percentage (default 0.03% or institutional proxy)
2. Securities Transaction Tax (STT):
   - 0.10% on Delivery Buy Turnover AND 0.10% on Delivery Sell Turnover
3. Exchange Transaction Charges:
   - 0.00345% on total turnover (NSE cash segment)
4. Goods & Services Tax (GST):
   - 18.0% applicable strictly on (Brokerage + Exchange Charges)
5. SEBI Turnover Charges:
   - Rs 10 per Crore of turnover (0.0001%)
6. Stamp Duty:
   - 0.015% applicable on Buy Turnover only (as per Indian Stamp Act)
7. Slippage Friction:
   - Configurable execution slippage (default 10 bps per side)
================================================================================
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from config import CONFIG, CostConfig


@dataclass(frozen=True)
class CostBreakdown:
    """Itemized audit record of all statutory and execution transaction costs."""
    buy_turnover: float
    sell_turnover: float
    total_turnover: float
    brokerage: float
    stt: float
    exchange_charges: float
    gst: float
    sebi_charges: float
    stamp_duty: float
    slippage_cost: float
    total_friction: float
    friction_bps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "buy_turnover": round(self.buy_turnover, 2),
            "sell_turnover": round(self.sell_turnover, 2),
            "total_turnover": round(self.total_turnover, 2),
            "brokerage": round(self.brokerage, 2),
            "stt": round(self.stt, 2),
            "exchange_charges": round(self.exchange_charges, 2),
            "gst": round(self.gst, 2),
            "sebi_charges": round(self.sebi_charges, 2),
            "stamp_duty": round(self.stamp_duty, 2),
            "slippage_cost": round(self.slippage_cost, 2),
            "total_friction": round(self.total_friction, 2),
            "friction_bps": round(self.friction_bps, 2)
        }


class NSETransactionCostCalculator:
    """Full-cycle cash delivery transaction cost and slippage calculator."""

    def __init__(self, cost_cfg: Optional[CostConfig] = None):
        self.cfg: CostConfig = cost_cfg or CONFIG.costs

    def calculate_turnover_costs(
        self,
        buy_price: float,
        sell_price: float,
        quantity: int
    ) -> CostBreakdown:
        """
        Calculates complete round-trip transaction costs for an entered and closed trade.

        Args:
            buy_price: Executed entry fill price per share.
            sell_price: Executed exit fill price per share.
            quantity: Number of shares traded.

        Returns:
            CostBreakdown dataclass with itemized fees in INR.
        """
        if quantity <= 0 or buy_price <= 0 or sell_price <= 0:
            return CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        buy_turnover = buy_price * quantity
        sell_turnover = sell_price * quantity
        total_turnover = buy_turnover + sell_turnover

        # 1. Brokerage (applied on total round-trip turnover)
        brokerage = total_turnover * self.cfg.brokerage_pct

        # 2. STT: 0.10% on Delivery Buy AND 0.10% on Delivery Sell
        stt_buy = buy_turnover * self.cfg.stt_delivery_buy_pct
        stt_sell = sell_turnover * self.cfg.stt_delivery_sell_pct
        stt_total = stt_buy + stt_sell

        # 3. Exchange Turnover Charges (0.00345%)
        exchange_charges = total_turnover * self.cfg.exchange_charges_pct

        # 4. GST: 18% on (Brokerage + Exchange Charges)
        gst = (brokerage + exchange_charges) * self.cfg.gst_pct

        # 5. SEBI Turnover Charges (Rs 10 per crore)
        sebi_charges = total_turnover * self.cfg.sebi_turnover_pct

        # 6. Stamp Duty: 0.015% on Buy Turnover only
        stamp_duty = buy_turnover * self.cfg.stamp_duty_buy_pct

        # 7. Slippage Cost (10 bps each side)
        slippage = total_turnover * self.cfg.estimated_slippage_pct

        total_friction = (
            brokerage +
            stt_total +
            exchange_charges +
            gst +
            sebi_charges +
            stamp_duty +
            slippage
        )

        # Friction in basis points relative to total round-trip turnover
        friction_bps = (total_friction / total_turnover) * 10000.0 if total_turnover > 0 else 0.0

        return CostBreakdown(
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            total_turnover=total_turnover,
            brokerage=brokerage,
            stt=stt_total,
            exchange_charges=exchange_charges,
            gst=gst,
            sebi_charges=sebi_charges,
            stamp_duty=stamp_duty,
            slippage_cost=slippage,
            total_friction=total_friction,
            friction_bps=friction_bps
        )

    def calculate_single_leg_cost(
        self,
        price: float,
        quantity: int,
        side: str = "BUY"
    ) -> float:
        """
        Calculates execution fees for a single trade leg (Buy or Sell).
        """
        if quantity <= 0 or price <= 0:
            return 0.0

        turnover = price * quantity
        brokerage = turnover * self.cfg.brokerage_pct
        exchange = turnover * self.cfg.exchange_charges_pct
        gst = (brokerage + exchange) * self.cfg.gst_pct
        sebi = turnover * self.cfg.sebi_turnover_pct
        slippage = turnover * self.cfg.estimated_slippage_pct

        if side.upper() == "BUY":
            stt = turnover * self.cfg.stt_delivery_buy_pct
            stamp = turnover * self.cfg.stamp_duty_buy_pct
        else:
            stt = turnover * self.cfg.stt_delivery_sell_pct
            stamp = 0.0

        return brokerage + stt + exchange + gst + sebi + stamp + slippage
