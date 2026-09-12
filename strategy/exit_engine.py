"""
NSE MOMENTUM 5™ — Engine B: Exit Intelligence & Position Management
================================================================================
Independent decision-support engine answering:
"I already own this stock. Should I HOLD, TRAIL, PARTIAL BOOK, or EXIT?"

Core Components:
1. Dynamic ATR Trailing Stops:
   Trailing Stop = Highest Price Since Entry - (ATR Multiplier * ATR 14)
2. Staged Profit Protection (Ratchet Mechanism):
   - Gain < +5%   : Normal trend incubation (2.0x - 2.5x ATR)
   - +5% to +10%  : Break-Even capital protection activated
   - +10% to +15% : Moderate trailing lock (2.0x ATR)
   - +15% to +20% : Aggressive trailing lock (1.5x ATR)
   - Gain > +20%  : Maximum profit lock (1.0x ATR tight trail)
3. Hold Score (0 to 100):
   Represents current health of existing trend structure:
   - 80 – 100 : STRONG HOLD
   - 65 – 79  : HOLD / TRAIL
   - 50 – 64  : WATCH / TRAIL
   - 35 – 49  : PARTIAL BOOK / EXIT WATCH
   - 0 – 34   : EXIT
4. Terminal Exit States:
   - HOLD
   - TRAIL
   - PARTIAL BOOK
   - EXIT
   - EMERGENCY EXIT (Hard stop violation)
================================================================================
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from config import CONFIG, ExitConfig
from utils.logger import setup_logger

logger = setup_logger("EXIT_INTELLIGENCE")


class ExitAction(str, Enum):
    """Primary position action recommendations."""
    HOLD = "HOLD"
    TRAIL = "TRAIL"
    PARTIAL_BOOK = "PARTIAL BOOK"
    EXIT = "EXIT"
    EMERGENCY_EXIT = "EMERGENCY EXIT"


@dataclass(frozen=True)
class ExitAssessment:
    """Institutional evaluation output for an existing holding."""
    symbol: str
    action: ExitAction
    hold_score: float                    # 0.0 to 100.0 Trend Health Score
    hold_category: str                   # 'STRONG HOLD', 'HOLD / TRAIL', etc.
    current_price: float
    unrealized_pnl_pct: float
    highest_price_since_entry: float
    active_trailing_stop: float
    hard_stop: float
    profit_tier: str
    primary_reasons: List[str] = field(default_factory=list)
    is_terminal: bool = False            # True = Must liquidate position immediately

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action.value,
            "hold_score": self.hold_score,
            "hold_category": self.hold_category,
            "current_price": round(self.current_price, 2),
            "unrealized_pnl_pct": round(self.unrealized_pnl_pct * 100.0, 2),
            "highest_price_since_entry": round(self.highest_price_since_entry, 2),
            "active_trailing_stop": round(self.active_trailing_stop, 2),
            "hard_stop": round(self.hard_stop, 2),
            "profit_tier": self.profit_tier,
            "primary_reasons": self.primary_reasons,
            "is_terminal": self.is_terminal
        }


class ExitIntelligenceEngine:
    """Evidence-driven position monitoring and exit management engine."""

    def __init__(self, exit_cfg: Optional[ExitConfig] = None):
        self.cfg: ExitConfig = exit_cfg or CONFIG.exit_cfg

    def evaluate_position(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        highest_price_since_entry: float,
        latest_features: Union[pd.Series, Dict[str, Any]],
        user_hard_stop: float = 0.0,
        user_target: float = 0.0,
        regime_permitted: bool = True
    ) -> ExitAssessment:
        """
        Evaluates an active position against trailing stops, profit ratchets, and trend health.

        Args:
            symbol: Ticker symbol.
            entry_price: Original fill price.
            current_price: Latest market price.
            highest_price_since_entry: High water mark price since trade inception.
            latest_features: Technical indicators series.
            user_hard_stop: Optional initial fixed stop loss price.
            user_target: Optional pre-defined upside target price.
            regime_permitted: Broad market health flag.

        Returns:
            ExitAssessment dataclass with actionable recommendations.
        """
        reasons: List[str] = []

        # Ensure valid high water mark
        peak_price = max(highest_price_since_entry, current_price, entry_price)
        gain_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
        peak_gain_pct = (peak_price - entry_price) / entry_price if entry_price > 0 else 0.0

        atr = float(latest_features.get("atr_14", current_price * 0.025) or (current_price * 0.025))
        ema10 = float(latest_features.get("ema_10", current_price) or current_price)
        ema20 = float(latest_features.get("ema_20", current_price) or current_price)
        rsi = float(latest_features.get("rsi_14", 50.0) or 50.0)
        vol_r5 = float(latest_features.get("vol_ratio_5d", 1.0) or 1.0)
        clv = float(latest_features.get("clv", 0.0) or 0.0)

        # ----------------------------------------------------------------------
        # 1. Staged Profit Protection Tier & Dynamic ATR Trailing Multiplier
        # ----------------------------------------------------------------------
        if peak_gain_pct >= self.cfg.tier4_gain_pct:
            profit_tier = "Tier 4: Super Gain (>=20%) — Tight 1.0x ATR Trailing"
            atr_mult = self.cfg.tier4_atr_multiplier
        elif peak_gain_pct >= self.cfg.tier3_gain_pct:
            profit_tier = "Tier 3: Strong Gain (15%-20%) — Aggressive 1.5x ATR Trailing"
            atr_mult = self.cfg.tier3_atr_multiplier
        elif peak_gain_pct >= self.cfg.tier2_gain_pct:
            profit_tier = "Tier 2: Moderate Gain (10%-15%) — 2.0x ATR Trailing Active"
            atr_mult = self.cfg.tier2_atr_multiplier
        elif peak_gain_pct >= self.cfg.tier1_gain_pct:
            profit_tier = "Tier 1: Initial Gain (5%-10%) — Break-Even / 2.5x ATR Trailing"
            atr_mult = self.cfg.tier1_atr_multiplier
        else:
            profit_tier = "Base Tier: Trend Incubation (<5%)"
            atr_mult = self.cfg.default_atr_multiplier

        # Calculate Dynamic Trailing Stop: Peak - (Multiplier * ATR)
        trailing_stop = peak_price - (atr_mult * atr)

        # In Tier 1 and above, ratchet trailing stop to guarantee at least break-even + fee buffer
        if peak_gain_pct >= self.cfg.tier1_gain_pct:
            break_even_floor = entry_price * 1.003  # Locks in trade costs
            trailing_stop = max(trailing_stop, break_even_floor)

        # Determine effective hard stop
        effective_hard_stop = user_hard_stop if user_hard_stop > 0 else (entry_price - (2.0 * atr))

        # ----------------------------------------------------------------------
        # 2. Hard Exit Violations (Emergency & Stop-Loss Breaches)
        # ----------------------------------------------------------------------
        if current_price <= effective_hard_stop:
            reasons.append(f"Hard Stop-Loss violated at INR {effective_hard_stop:.2f}")
            return ExitAssessment(
                symbol=symbol,
                action=ExitAction.EMERGENCY_EXIT,
                hold_score=0.0,
                hold_category="EXIT",
                current_price=current_price,
                unrealized_pnl_pct=gain_pct,
                highest_price_since_entry=peak_price,
                active_trailing_stop=trailing_stop,
                hard_stop=effective_hard_stop,
                profit_tier=profit_tier,
                primary_reasons=reasons,
                is_terminal=True
            )

        if current_price <= trailing_stop:
            reasons.append(
                f"Trailing Stop breached at INR {trailing_stop:.2f} "
                f"(Locked in from high of INR {peak_price:.2f})"
            )
            return ExitAssessment(
                symbol=symbol,
                action=ExitAction.EXIT,
                hold_score=15.0,
                hold_category="EXIT",
                current_price=current_price,
                unrealized_pnl_pct=gain_pct,
                highest_price_since_entry=peak_price,
                active_trailing_stop=trailing_stop,
                hard_stop=effective_hard_stop,
                profit_tier=profit_tier,
                primary_reasons=reasons,
                is_terminal=True
            )

        # ----------------------------------------------------------------------
        # 3. Hold Score (0 to 100) — Trend Health Evaluation
        # ----------------------------------------------------------------------
        hold_score = 50.0  # Baseline neutral

        # A. Moving Average Support (Trend Structure)
        if current_price > ema10:
            hold_score += 20.0
            reasons.append("Holding firmly above rising EMA 10")
        elif current_price > ema20:
            hold_score += 10.0
            reasons.append("Resting on 20-day trend support (EMA 20)")
        else:
            hold_score -= 20.0
            reasons.append("Broken below EMA 20 support (Trend deterioration)")

        # B. RSI Momentum Health
        if 55.0 <= rsi <= 75.0:
            hold_score += 15.0
        elif rsi < 45.0:
            hold_score -= 15.0
            reasons.append(f"RSI collapsed below 45 ({rsi:.1f})")
        elif rsi > 85.0:
            hold_score -= 5.0
            reasons.append("Overbought RSI (>85); minor pullback risk")

        # C. Volume Behavior (Institutional Distribution Check)
        if vol_r5 >= 2.0 and clv <= -0.40:
            hold_score -= 25.0
            reasons.append("Distribution alert: Heavy volume sell-off closing at session lows")

        # D. Macro Regime Environment
        if not regime_permitted:
            hold_score -= 10.0
            reasons.append("Macro Market Regime is BEARISH; elevated risk")
        else:
            hold_score += 10.0

        # E. Upside Target Proximity Check
        if user_target > 0 and current_price >= user_target:
            hold_score -= 10.0
            reasons.append(f"Upside target reached at INR {user_target:.2f}")

        hold_score = round(float(np.clip(hold_score, 0.0, 100.0)), 2)

        # ----------------------------------------------------------------------
        # 4. Synthesize Action Recommendation
        # ----------------------------------------------------------------------
        if hold_score >= self.cfg.hold_score_strong_hold:
            action = ExitAction.HOLD
            category = "STRONG HOLD"
            reasons.append("Trend health robust. Maintain position.")
            is_terminal = False

        elif hold_score >= self.cfg.hold_score_trail:
            action = ExitAction.TRAIL
            category = "HOLD / TRAIL"
            reasons.append("Trend intact. Maintain position with trailing protection.")
            is_terminal = False

        elif hold_score >= self.cfg.hold_score_watch:
            action = ExitAction.TRAIL
            category = "WATCH / TRAIL"
            reasons.append("Momentum softening. Tighten trailing stops.")
            is_terminal = False

        elif hold_score >= self.cfg.hold_score_partial_book:
            action = ExitAction.PARTIAL_BOOK
            category = "PARTIAL BOOK / EXIT WATCH"
            reasons.append("Deteriorating trend health. Recommend booking 50% profit.")
            is_terminal = False

        else:
            action = ExitAction.EXIT
            category = "EXIT"
            reasons.append("Hold score broken below minimum acceptable threshold. Full exit advised.")
            is_terminal = True

        return ExitAssessment(
            symbol=symbol,
            action=action,
            hold_score=hold_score,
            hold_category=category,
            current_price=current_price,
            unrealized_pnl_pct=gain_pct,
            highest_price_since_entry=peak_price,
            active_trailing_stop=trailing_stop,
            hard_stop=effective_hard_stop,
            profit_tier=profit_tier,
            primary_reasons=reasons,
            is_terminal=is_terminal
        )
