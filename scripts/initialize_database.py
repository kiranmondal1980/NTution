"""
NSE MOMENTUM 5™ — CLI Tool: Database Initialization & Schema Verification
================================================================================
Command-line utility to initialize the relational SQLite schema, verify tables,
and seed the default liquid NSE universe and runtime configuration.

Usage:
    python scripts/initialize_database.py
    python scripts/initialize_database.py --reset (Drops and recreates all tables)
================================================================================
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path to allow direct CLI execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.initialize import init_database
from database.db import engine, Base
from utils.logger import setup_logger

logger = setup_logger("CLI_DB_INIT")


def main() -> int:
    parser = argparse.ArgumentParser(description="NSE MOMENTUM 5™ Database Initializer")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="CAUTION: Drops all existing tables before re-creating schema and seeding."
    )
    args = parser.parse_args()

    if args.reset:
        logger.warning("Reset requested: Dropping all database tables...")
        try:
            Base.metadata.drop_all(bind=engine)
            logger.info("All existing tables dropped successfully.")
        except Exception as exc:
            logger.critical(f"Failed to drop tables: {exc}")
            return 1

    logger.info("Starting database schema creation and universe seeding...")
    success = init_database()

    if success:
        logger.info("SUCCESS: Database schema initialized and universe seeded successfully.")
        return 0
    else:
        logger.error("FAILURE: Database initialization encountered errors.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
