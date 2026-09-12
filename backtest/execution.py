"""
NSE MOMENTUM 5™ — Order Execution & Realistic Slippage Simulator
================================================================================
Simulates realistic next-day execution fills with slippage, overnight gap handling,
and intraday stop/target chronology.

STRICT CHRONOLOGY RULE (MANDATORY ANTI-LOOKAHEAD ENFORCEMENT):
If a signal is generated at the close of Session T, the earliest possible entry
execution occurs on Session T+1 at the Market Open.
Under NO circumstances is same-bar execution at Close[T] permitted.

Fill Realism Logic:
1. Market Open Orders:
   - BUY fills at: Open[T+1] * (1.0 + Slippage)
   - SELL fills at: Open[T+1] * (1.0 - Slippage)
2. Overnight Gap-Down Stop Handling:
   - If Open[T+1] opens BELOW the stop-loss price, the fill occurs at Open[T+1]
     (slippage-adjusted), correctly modeling overnight gap loss.
3. Intraday Stop Executions:
   - If Low[T+1] <= Stop Loss Price, fills at: Stop Price * (1.0 - Slippage)
4. Intraday Target Executions:
   - If High[T+1] >= Target Price, fills at: Target Price * (1.0 - Slippage)
================================================================================
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
from config import CONFIG, CostConfig
from utils.logger import setup_logger

logger = setup_logger("EXECUTION_SIMULATOR")


@dataclass(frozen=True)
class SimulatedOrder:
    """Audit record of a simulated executed order."""
    symbol: str
    signal_date: datetime
    execution_date: datetime
    side: str                     # "BUY" or "SELL"
    order_type: str               # "MARKET_OPEN", "STOP_LOSS", "TRAILING_STOP", "TARGET"
    requested_price: float
    executed_price: float
    quantity: int
    slippage_inr: float
    execution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_date": str(self.signal_date),
            "execution_date": str(self.execution_date),
            "side": self.side,
            "order_type": self.order_type,
            "requested_price": round(self.requested_price, 2),
            "executed_price": round(self.executed_price, 2),
            "quantity": self.quantity,
            "slippage_inr": round(self.slippage_inr, 2),
            "execution_notes": self.execution_notes
        }


class ExecutionEngine:
    """Simulates market order executions incorporating slippage and price boundaries."""

    def __init__(self, cost_cfg: Optional[CostConfig] = None):
        self.cfg: CostConfig = cost_cfg or CONFIG.costs

    def execute_market_open(
        self,
        symbol: str,
        signal_date: datetime,
        next_date: datetime,
        next_open_price: float,
        quantity: int,
        side: str = "BUY"
    ) -> SimulatedOrder:
        """
        Simulates entry or exit execution at next session open with slippage applied.

        Args:
            symbol: Ticker symbol.
            signal_date: Timestamp of session T when signal was generated.
            next_date: Timestamp of session T+1 when order executes.
            next_open_price: Open price of session T+1.
            quantity: Number of shares.
            side: "BUY" or "SELL".

        Returns:
            SimulatedOrder with realistic filled price.
        """
        slip_pct = self.cfg.estimated_slippage_pct
        if side.upper() == "BUY":
            fill_price = next_open_price * (1.0 + slip_pct)
        else:
            fill_price = next_open_price * (1.0 - slip_pct)

        total_slippage_inr = abs(fill_price - next_open_price) * quantity

        return SimulatedOrder(
            symbol=symbol,
            signal_date=signal_date,
            execution_date=next_date,
            side=side.upper(),
            order_type="MARKET_OPEN",
            requested_price=next_open_price,
            executed_price=fill_price,
            quantity=quantity,
            slippage_inr=total_slippage_inr,
            execution_notes=f"Filled next session open with {slip_pct*10000:.0f} bps slippage."
        )

    def execute_stop_exit(
        self,
        symbol: str,
        execution_date: datetime,
        stop_price: float,
        bar_open: float,
        bar_low: float,
        quantity: int,
        is_trailing: bool = False
    ) -> Optional[SimulatedOrder]:
        """
        Simulates stop execution handling gap-downs below stop price.

        Args:
            symbol: Ticker symbol.
            execution_date: Date of execution.
            stop_price: Effective stop loss price.
            bar_open: Today's Open price.
            bar_low: Today's Low price.
            quantity: Number of shares.
            is_trailing: True if triggered by dynamic trailing stop.

        Returns:
            SimulatedOrder if stop was triggered, otherwise None.
        """
        # Stop is triggered if price dropped to or below stop_price
        if bar_low > stop_price and bar_open > stop_price:
            return None

        slip_pct = self.cfg.estimated_slippage_pct

        # Gap down handling: If Open opened below Stop, filled at Open
        if bar_open < stop_price:
            base_fill = bar_open
            note = "Overnight gap-down below stop price: Executed at market open."
        else:
            base_fill = stop_price
            note = "Intraday stop breach executed."

        fill_price = base_fill * (1.0 - slip_pct)
        total_slippage_inr = (base_fill - fill_price) * quantity
        order_type = "TRAILING_STOP" if is_trailing else "STOP_LOSS"

        return SimulatedOrder(
            symbol=symbol,
            signal_date=execution_date,
            execution_date=execution_date,
            side="SELL",
            order_type=order_type,
            requested_price=stop_price,
            executed_price=fill_price,
            quantity=quantity,
            slippage_inr=total_slippage_inr,
            execution_notes=note
        )

    def execute_target_exit(
        self,
        symbol: str,
        execution_date: datetime,
        target_price: float,
        bar_open: float,
        bar_high: float,
        quantity: int
    ) -> Optional[SimulatedOrder]:
        """
        Simulates profit target execution handling gap-ups above target price.
        """
        if bar_high < target_price and bar_open < target_price:
            return None

        slip_pct = self.cfg.estimated_slippage_pct

        # Gap up handling: If Open opened above Target, filled at Open
        if bar_open > target_price:
            base_fill = bar_open
            note = "Gap-up above target: Executed at market open."
        else:
            base_fill = target_price
            note = "Intraday target reached."

        fill_price = base_fill * (1.0 - slip_pct)
        total_slippage_inr = (base_fill - fill_price) * quantity

        return SimulatedOrder(
            symbol=symbol,
            signal_date=execution_date,
            execution_date=execution_date,
            side="SELL",
            order_type="TARGET",
            requested_price=target_price,
            executed_price=fill_price,
            quantity=quantity,
            slippage_inr=total_slippage_inr,
            execution_notes=note
        )
