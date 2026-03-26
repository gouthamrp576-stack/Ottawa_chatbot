"""Logging helpers for the Ottawa Newcomer Assistant."""

from __future__ import annotations

import logging


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging once for the current process."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=getattr(logging, level, logging.INFO), format=_LOG_FORMAT)
        return

    root_logger.setLevel(getattr(logging, level, logging.INFO))
    for handler in root_logger.handlers:
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))


def get_logger(name: str) -> logging.Logger:
    """Return a module logger."""
    return logging.getLogger(name)
