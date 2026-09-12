"""
NSE MOMENTUM 5™ — SQLAlchemy Database Engine & Session Management
================================================================================
Provides thread-safe relational database connectivity, connection pooling,
and transactional context management for SQLite (and compatible SQL engines).

Features:
- Thread-safe session generator with automatic commit and rollback
- SQLite performance tuning (WAL mode, busy timeout, foreign key enforcement)
- Connection verification and database health-check utilities
================================================================================
"""

from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from config import CONFIG
from utils.logger import setup_logger

logger = setup_logger("DATABASE_ENGINE")

# Base class for all ORM model definitions
Base = declarative_base()

# Initialize Engine using master configuration
engine = create_engine(
    CONFIG.db.db_uri,
    echo=CONFIG.db.echo_sql,
    pool_pre_ping=CONFIG.db.pool_pre_ping,
    connect_args=CONFIG.db.connect_args if "sqlite" in CONFIG.db.db_uri else {}
)


# Optimize SQLite for concurrent reading and writing
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Applies high-concurrency pragmas for SQLite databases:
    - WAL (Write-Ahead Logging) enables non-blocking concurrent reads & writes
    - synchronous = NORMAL maintains ACID safety while maximizing speed
    - foreign_keys = ON enforces referential integrity
    - busy_timeout = 30000ms prevents lock contention errors
    """
    if "sqlite" in CONFIG.db.db_uri:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.execute("PRAGMA busy_timeout=30000;")
        except Exception as e:
            logger.warning(f"Could not apply SQLite performance PRAGMAs: {e}")
        finally:
            cursor.close()


# Session Factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager for atomic transactional database operations.
    
    Usage:
        with get_db_session() as session:
            session.add(record)
            # Commit happens automatically on clean exit
            # Rollback happens automatically if any exception is raised
    """
    session: Session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Database transaction failed and was rolled back: {str(exc)}")
        raise
    finally:
        session.close()


def check_db_connection() -> bool:
    """
    Verifies that the database engine can successfully connect and execute queries.

    Returns:
        True if connected successfully, False otherwise.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        return True
    except Exception as e:
        logger.critical(f"Database connection verification failed: {str(e)}")
        return False
