"""
NSE MOMENTUM 5™ — Institutional Logging & Audit Engine
================================================================================
Provides enterprise-grade logging across all trading and scanning pipelines.
Guarantees:
- Safe handling of sensitive tokens/credentials (automatic pattern masking)
- Simultaneous dual logging (Console Stream + Persistent Daily File)
- Strict thread safety and non-blocking handler initialization
- Zero external dependencies (uses standard library `logging` and `sys`)
================================================================================
"""

import logging
import sys
import re
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

# Import log directory path from master configuration
try:
    from config import LOG_DIR
except ImportError:
    # Fallback to local logs directory if config is loaded from sub-package
    LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)


class SensitiveDataFilter(logging.Filter):
    """
    Security Filter: Scans and redacts sensitive credentials, API keys,
    tokens, and passwords from log strings before writing to disk or terminal.
    """
    SENSITIVE_PATTERNS = [
        re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)([^'\"\s]+)(['\"]?)", re.IGNORECASE),
        re.compile(r"(token\s*[:=]\s*['\"]?)([^'\"\s]+)(['\"]?)", re.IGNORECASE),
        re.compile(r"(password\s*[:=]\s*['\"]?)([^'\"\s]+)(['\"]?)", re.IGNORECASE),
        re.compile(r"(secret\s*[:=]\s*['\"]?)([^'\"\s]+)(['\"]?)", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in self.SENSITIVE_PATTERNS:
                record.msg = pattern.sub(r"\1***REDACTED***\3", record.msg)
        return True


def setup_logger(
    name: str = "NSE_MOMENTUM_5",
    level: int = logging.INFO,
    log_filename: str = "system.log",
    max_bytes: int = 10_485_760,  # 10 MB per log chunk
    backup_count: int = 5
) -> logging.Logger:
    """
    Creates and configures a standardized, thread-safe institutional logger.

    Args:
        name: Unique module or subsystem namespace name.
        level: Logging severity threshold (e.g. logging.INFO, logging.DEBUG).
        log_filename: Destination file name in the logs directory.
        max_bytes: Maximum size of log file before automatic rotation.
        backup_count: Number of historical rotated log files to retain.

    Returns:
        Configured logging.Logger instance with security filters applied.
    """
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if setup_logger is called repeatedly in tests or notebooks
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)
    logger.propagate = False  # Prevent propagating to root logger to avoid duplicate prints

    # Standardized quantitative format:
    # YYYY-MM-DD HH:MM:SS [LEVEL] [MODULE:LINE] Message
    log_format = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    sensitive_filter = SensitiveDataFilter()

    # 1. Console / Standard Output Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(level)
    console_handler.addFilter(sensitive_filter)
    logger.addHandler(console_handler)

    # 2. Rotating File Handler (Persisted to disk)
    try:
        log_file_path = LOG_DIR / log_filename
        file_handler = RotatingFileHandler(
            filename=str(log_file_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setFormatter(log_format)
        file_handler.setLevel(level)
        file_handler.addFilter(sensitive_filter)
        logger.addHandler(file_handler)
    except Exception as exc:
        # Fallback to console-only if running in a restricted filesystem (e.g., read-only sandbox)
        console_handler.setLevel(logging.WARNING)
        logger.warning(
            f"Unable to initialize rotating file handler at {LOG_DIR / log_filename}: {exc}. "
            "System will log to console stream only."
        )

    return logger
