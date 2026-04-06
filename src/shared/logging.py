"""Structured JSON logging with correlation ID support."""

from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

# Context variable holding the current request's correlation ID.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")


class CorrelationIdFilter(logging.Filter):
    """Inject ``correlation_id`` from *contextvars* into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id_var.get("")  # type: ignore[attr-defined]
        return True


class JSONFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", ""),
        }
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(service_name: str, log_level: str = "INFO") -> None:
    """Configure the root logger with JSON output and correlation-ID filter.

    Should be called once at service startup.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls (e.g. tests).
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        handler.addFilter(CorrelationIdFilter())
        root.addHandler(handler)

    # Tag root logger so downstream loggers inherit the service name.
    logging.getLogger(service_name)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger that inherits the root JSON configuration."""
    return logging.getLogger(name)
