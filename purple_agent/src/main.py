"""
Application Entry Point -- Purple Agent v7.0

This is the explicit entry point for the Purple Agent application.
It handles:
1. Loading environment variables from .env
2. Validating PYTHONHASHSEED for determinism
3. Configuring logging with RedactingFormatter (credential protection)
4. Starting the FastAPI/uvicorn server

Usage:
    python src/main.py           # Direct execution
    uvicorn src.main:app         # Via uvicorn (used in Dockerfile)
"""
import logging
import os
import re
import sys

from dotenv import load_dotenv

# Load .env BEFORE any other imports that read os.getenv()
load_dotenv()

from src.config import (
    A2A_SERVER_HOST,
    A2A_SERVER_PORT,
    AGENT_VERSION,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_REDACT_PATTERNS,
)


class RedactingFormatter(logging.Formatter):
    """Formatter that redacts sensitive patterns from log output.

    Prevents accidental leakage of API keys, tokens, and passwords
    into Docker logs, Splunk, Datadog, or any log aggregator.
    """

    def __init__(self, fmt: str | None = None, redact_patterns: list[str] | None = None) -> None:
        super().__init__(fmt)
        self._patterns = [
            re.compile(p) for p in (redact_patterns or LOG_REDACT_PATTERNS)
        ]

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        for pattern in self._patterns:
            message = pattern.sub("[REDACTED]", message)
        return message


def configure_logging() -> None:
    """Configure root logger with RedactingFormatter."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(RedactingFormatter(LOG_FORMAT))
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )


def validate_determinism() -> None:
    """Warn if PYTHONHASHSEED is not set to 0.

    Deterministic output requires PYTHONHASHSEED=0 so that dict/set
    iteration order is reproducible across runs. This is NOT a hard
    failure because local dev may not set it, but production/Docker MUST.
    """
    hashseed = os.environ.get("PYTHONHASHSEED", "")
    if hashseed != "0":
        logging.getLogger(__name__).warning(
            "PYTHONHASHSEED=%s (expected '0' for deterministic output). "
            "Set PYTHONHASHSEED=0 in environment or Dockerfile.",
            hashseed or "<unset>",
        )


# Configure logging immediately
configure_logging()
validate_determinism()

logger = logging.getLogger(__name__)


def create_app():
    """Create and return the FastAPI application.

    Deferred import to ensure logging is configured first and
    to avoid circular imports during testing.
    """
    from src.core.a2a_server import app as _app  # noqa: F811
    return _app


# Module-level app for uvicorn: `uvicorn src.main:app`
app = create_app()


if __name__ == "__main__":
    import uvicorn

    logger.info(
        "Starting Purple Agent v%s on %s:%d",
        AGENT_VERSION,
        A2A_SERVER_HOST,
        A2A_SERVER_PORT,
    )
    uvicorn.run(
        "src.main:app",
        host=A2A_SERVER_HOST,
        port=A2A_SERVER_PORT,
        log_level=LOG_LEVEL.lower(),
    )
