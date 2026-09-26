"""Structured JSON logging for the API."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

RESERVED_ATTRS = (
    "request_id",
    "method",
    "path",
    "status",
    "duration_ms",
    "endpoint",
    "backend",
)


class JsonFormatter(logging.Formatter):
    """One JSON object per log line; extra record attributes are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for attr in RESERVED_ATTRS:
            if hasattr(record, attr):
                payload[attr] = getattr(record, attr)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "info") -> None:
    """Route uvicorn + app loggers through a single JSON handler."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True
    # The request-logging middleware emits the structured access line.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
