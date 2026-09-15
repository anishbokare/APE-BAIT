"""
APE-BAIT Structured Logger
JSON-structured logging with rotation, console rich output,
and performance metric tracking.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.logging import RichHandler


_console = Console(highlight=True)


class StructuredLogger:
    """
    Structured logger that emits JSON-formatted log records to file
    and rich-formatted output to console.
    """

    def __init__(
        self,
        name: str,
        log_file: str = "logs/ape-bait.log",
        level: str = "INFO",
        max_size_mb: int = 100,
        backup_count: int = 5,
    ):
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.handlers.clear()

        # ── Console Handler (Rich) ──────────────────────────────
        console_handler = RichHandler(
            console=_console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.addHandler(console_handler)

        # ── File Handler (Rotating JSON) ────────────────────────
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        file_handler.setLevel(logging.DEBUG)
        self._logger.addHandler(file_handler)

    def _log(self, level: int, msg: str, **kwargs: Any) -> None:
        extra = {"extra_fields": kwargs}
        self._logger.log(level, msg, extra=extra)

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, msg, **kwargs)

    def metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Log a performance metric."""
        self._log(
            logging.INFO,
            f"METRIC {name}={value}",
            metric_name=name,
            metric_value=value,
            tags=tags or {},
        )


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach extra structured fields
        extra = getattr(record, "extra_fields", {})
        if extra:
            log_obj.update(extra)

        # Attach exception info if present
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


class TimingContext:
    """Context manager for measuring execution time."""

    def __init__(self, logger: StructuredLogger, operation: str):
        self._logger = logger
        self._operation = operation
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._logger.metric(
            f"latency.{self._operation}",
            round(elapsed_ms, 3),
            tags={"unit": "ms"},
        )

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000


# ── Module-level convenience logger ─────────────────────────────
_default_logger: Optional[StructuredLogger] = None


def get_logger(
    name: str = "ape-bait",
    log_file: str = "logs/ape-bait.log",
    level: str = "INFO",
) -> StructuredLogger:
    """Get or create a named logger."""
    global _default_logger
    if _default_logger is None or _default_logger.name != name:
        _default_logger = StructuredLogger(name, log_file=log_file, level=level)
    return _default_logger


def timing(logger: StructuredLogger, operation: str) -> TimingContext:
    """Convenience function to create a timing context."""
    return TimingContext(logger, operation)
