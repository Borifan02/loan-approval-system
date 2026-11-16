"""Utility functions and logging helpers for the Loan Approval system."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict


def setup_logger(name: str = "loan_approval") -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def format_currency(value: float, decimals: int = 2) -> str:
    """Formats a float as standard USD currency."""
    return f"${value:,.{decimals}f}"


def safe_json_dumps(data: Dict[str, Any], indent: int = 2) -> str:
    """Serializes a dictionary to pretty JSON, converting non-standard types."""
    def default_serializer(obj: Any) -> Any:
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    return json.dumps(data, indent=indent, default=default_serializer)
