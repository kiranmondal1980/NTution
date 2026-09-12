"""
NSE MOMENTUM 5™ — Active Holdings Management Engine
================================================================================
Manages persistent user swing holdings for Engine B (Exit Intelligence).
Provides atomic database CRUD operations, high-water-mark peak price tracking,
dynamic stop updates, and partial booking / liquidation workflows.
================================================================================
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from database.db import get_db_session
from database.models import PositionRecord
from utils.logger import setup_logger

logger = setup_logger("HOLDINGS_MANAGER")


class HoldingsManager:
    """Manages active position state, trailing stops, and liquidations in SQLite."""

    @staticmethod
    def add_position(
        symbol: str,
        entry_date: datetime,
        entry_price: float,
        quantity: int,
        stop_loss: float,
        target_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Registers a new swing trade holding into persistent storage.

        Args:
            symbol: Ticker symbol (e.g., 'TRENT.NS').
            entry_date: Timestamp of entry.
            entry_price: Executed fill price.
            quantity: Total shares purchased.
            stop_loss: Initial technical stop loss price.
            target_price: Optional initial profit target.

        Returns:
            Dictionary representing the created position record.
        """
        clean_sym = symbol.strip().upper()
        if not clean_sym.endswith(".NS") and not clean_sym.startswith("^"):
            clean_sym = f"{clean_sym}.NS"

        with get_db_session() as session:
            pos = PositionRecord(
                symbol=clean_sym,
                entry_date=entry_date,
                entry_price=entry_price,
                quantity=quantity,
                current_stop=stop_loss,
                target_price=target_price,
                highest_price_since_entry=entry_price,
                status="OPEN",
                hold_score=80.0,
                exit_action="HOLD",
                last_updated=datetime.utcnow()
            )
            session.add(pos)
            session.flush()

            pos_dict = {
                "id": pos.id,
                "symbol": pos.symbol,
                "entry_date": pos.entry_date,
                "entry_price": pos.entry_price,
                "quantity": pos.quantity,
                "current_stop": pos.current_stop,
                "target_price": pos.target_price,
                "highest_price_since_entry": pos.highest_price_since_entry,
                "status": pos.status
            }
            session.commit()

        logger.info(
            f"Added active holding: {clean_sym} ({quantity} shares @ INR {entry_price:.2f}, Stop: {stop_loss:.2f})"
        )
        return pos_dict

    @staticmethod
    def get_open_positions() -> List[Dict[str, Any]]:
        """
        Retrieves all active open positions from the database.

        Returns:
            List of position dictionaries.
        """
        with get_db_session() as session:
            records = session.query(PositionRecord).filter(
                PositionRecord.status == "OPEN"
            ).order_by(PositionRecord.entry_date.desc()).all()

            positions = []
            for r in records:
                positions.append({
                    "id": r.id,
                    "symbol": r.symbol,
                    "entry_date": r.entry_date,
                    "entry_price": r.entry_price,
                    "quantity": r.quantity,
                    "current_stop": r.current_stop,
                    "target_price": r.target_price,
                    "highest_price_since_entry": r.highest_price_since_entry,
                    "status": r.status,
                    "exit_action": r.exit_action,
                    "hold_score": r.hold_score,
                    "last_updated": r.last_updated
                })
            return positions

    @staticmethod
    def update_position_hwm(
        position_id: int,
        latest_price: float,
        active_trailing_stop: float,
        hold_score: Optional[float] = None,
        exit_action: Optional[str] = None
    ) -> None:
        """
        Ratchets the high-water-mark price and dynamic trailing stop upward.
        Never moves trailing stops downward.
        """
        with get_db_session() as session:
            pos = session.query(PositionRecord).filter(PositionRecord.id == position_id).first()
            if pos:
                # Ratchet peak high upward
                pos.highest_price_since_entry = max(pos.highest_price_since_entry, latest_price)
                # Ratchet trailing stop upward only
                pos.current_stop = max(pos.current_stop, active_trailing_stop)

                if hold_score is not None:
                    pos.hold_score = hold_score
                if exit_action is not None:
                    pos.exit_action = exit_action

                pos.last_updated = datetime.utcnow()
                session.commit()

    @staticmethod
    def partial_book_position(
        position_id: int,
        shares_to_book: int
    ) -> Optional[int]:
        """
        Reduces position share size (e.g. 50% partial booking) while keeping position open.

        Returns:
            Remaining share count, or None if position not found.
        """
        with get_db_session() as session:
            pos = session.query(PositionRecord).filter(PositionRecord.id == position_id).first()
            if not pos:
                return None

            if shares_to_book >= pos.quantity:
                # Close fully if booking equals or exceeds total quantity
                pos.status = "CLOSED"
                pos.quantity = 0
            else:
                pos.quantity -= shares_to_book

            pos.last_updated = datetime.utcnow()
            rem_qty = pos.quantity
            session.commit()

            logger.info(f"Position {position_id} partially booked: {shares_to_book} shares sold. Remaining: {rem_qty}")
            return rem_qty

    @staticmethod
    def close_position(position_id: int) -> bool:
        """
        Marks an open position as CLOSED.

        Returns:
            True if position was found and closed, False otherwise.
        """
        with get_db_session() as session:
            pos = session.query(PositionRecord).filter(PositionRecord.id == position_id).first()
            if pos:
                pos.status = "CLOSED"
                pos.last_updated = datetime.utcnow()
                session.commit()
                logger.info(f"Position {position_id} ({pos.symbol}) marked CLOSED.")
                return True
            return False
