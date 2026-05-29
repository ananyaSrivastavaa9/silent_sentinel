"""
src/utils/logger.py
===================
Silent Sentinel — Edge AI Research Platform
Structured JSON logging setup — zero external dependencies.

Design rationale
----------------
``structlog`` is an excellent library, but adding it to a constrained edge
deployment introduces a non-trivial dependency tree.  This module achieves
the same observable behaviour (machine-parseable JSON log lines, consistent
field names, per-record timestamps) using only the Python standard library:

* ``logging.Formatter`` subclass (``_JSONFormatter``) serialises every
  ``LogRecord`` to a single-line JSON object.
* ``logging.Filter`` subclass (``_ContextFilter``) injects a static
  ``service`` tag so that log aggregators (Fluentd, Loki, Vector) can route
  records without parsing the message body.
* ``setup_logging`` is **idempotent**: repeated calls (e.g. during hot-reload
  in tests) replace handlers rather than stacking duplicates.

Log record schema
-----------------
Every emitted line is a JSON object with the following guaranteed keys:

.. code-block:: json

    {
        "timestamp": "2024-07-15T12:34:56.789012Z",
        "level":     "INFO",
        "logger":    "src.core.event_bus",
        "service":   "silent-sentinel",
        "message":   "Publishing event.",
        "event_type": "AcousticAnomalyEvent",
        "listener_count": 2
    }

Keys beyond ``timestamp``, ``level``, ``logger``, ``service``, and
``message`` are sourced from the ``extra={…}`` dict passed to each log call.

Usage
-----
>>> from src.utils.logger import setup_logging, get_logger
>>> setup_logging(level="DEBUG")
>>> log = get_logger(__name__)
>>> log.info("System ready.", extra={"component": "main"})
{"timestamp": "...", "level": "INFO", "logger": "...", ...}
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERVICE_NAME = "silent-sentinel"

# Fields that already live as top-level keys in our schema; they must not be
# duplicated when we drain ``extra`` from the LogRecord.
_RESERVED_KEYS: frozenset[str] = frozenset(
    {
        "timestamp",
        "level",
        "logger",
        "service",
        "message",
        # Standard LogRecord attributes we do NOT want to surface:
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }
)


# ===========================================================================
# Custom formatter
# ===========================================================================


class _JSONFormatter(logging.Formatter):
    """Render a ``LogRecord`` as a single-line JSON object.

    The output schema is stable and documented at the module level.
    Exception info (``exc_info``) is serialised under the ``"exception"`` key
    as a plain string to remain JSON-safe.

    Parameters
    ----------
    service:
        Static service tag injected into every record.
    """

    def __init__(self, service: str = _SERVICE_NAME) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Serialise *record* to a JSON string."""
        # Core schema — always present.
        payload: dict[str, Any] = {
            "timestamp": self._iso_utc(record.created),
            "level": record.levelname,
            "logger": record.name,
            "service": self._service,
            "message": record.getMessage(),
        }

        # Drain caller-supplied ``extra`` fields.
        for key, value in record.__dict__.items():
            if key not in _RESERVED_KEYS:
                payload[key] = value

        # Append exception traceback when present.
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Append stack info when present (Python 3.2+).
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)

    @staticmethod
    def _iso_utc(epoch: float) -> str:
        """Convert a Unix epoch float to an ISO-8601 UTC string."""
        return (
            datetime.fromtimestamp(epoch, tz=timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )


# ===========================================================================
# Public API
# ===========================================================================

_VALID_LEVELS: frozenset[str] = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
)


def setup_logging(
    level: str = "INFO",
    service: str = _SERVICE_NAME,
    stream: Any = sys.stdout,
) -> None:
    """Configure the root logger to emit structured JSON to *stream*.

    This function is **idempotent**: if a ``_JSONFormatter`` handler is already
    attached to the root logger it is replaced rather than duplicated, so
    calling ``setup_logging`` multiple times (common in test suites) is safe.

    Parameters
    ----------
    level:
        Minimum log level.  Case-insensitive; must be one of
        ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``, ``CRITICAL``.
    service:
        Static ``"service"`` tag written into every log record.  Override
        when running multiple Silent Sentinel micro-services in the same
        log aggregation pipeline.
    stream:
        Output stream for log records.  Defaults to ``sys.stdout`` so that
        container runtimes (Docker, containerd) capture logs via standard
        stream redirection.  Pass ``sys.stderr`` for error-only side-channels.

    Raises
    ------
    ValueError
        If *level* is not one of the accepted literals.
    """
    normalised = level.upper()
    if normalised not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level {level!r}. "
            f"Must be one of: {', '.join(sorted(_VALID_LEVELS))}."
        )

    numeric_level = logging.getLevelName(normalised)
    formatter = _JSONFormatter(service=service)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove any existing handlers that use our formatter to stay idempotent.
    root.handlers = [
        h for h in root.handlers if not isinstance(h.formatter, _JSONFormatter)
    ]

    handler = logging.StreamHandler(stream)
    handler.setLevel(numeric_level)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Silence noisy third-party loggers at WARNING by default.
    for noisy in ("urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger bound to the configured JSON pipeline.

    This is a thin convenience wrapper around ``logging.getLogger`` that
    serves as the single call-site in application code, making it easy to
    swap the underlying implementation later (e.g. to ``structlog``) without
    touching every import.

    Parameters
    ----------
    name:
        Logger name; conventionally ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Standard library logger configured by ``setup_logging``.

    Examples
    --------
    >>> log = get_logger(__name__)
    >>> log.info("Sensor initialised.", extra={"sensor": "IMU-1"})
    """
    return logging.getLogger(name)


# ===========================================================================
# Smoke-test entry point
# ===========================================================================

if __name__ == "__main__":
    setup_logging(level="DEBUG")
    log = get_logger(__name__)

    log.debug("Debug message — verbose tracing.", extra={"phase": "init"})
    log.info("Service starting.", extra={"version": "0.1.0"})
    log.warning("Sensor calibration overdue.", extra={"sensor_id": "IMU-0"})
    log.error("Feature extraction failed.", extra={"frame_id": 42})

    try:
        raise RuntimeError("Simulated pipeline fault.")
    except RuntimeError:
        log.exception(
            "Unhandled exception in acoustic pipeline.",
            extra={"component": "acoustic"},
        )

    log.info("Smoke test complete.")