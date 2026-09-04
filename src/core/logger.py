"""Centralised logging for PARAKH.

Stage1.md sec.4 requires that no bad row ever crashes the pipeline and that
every error is logged. This module gives every component a logger that writes
to both stderr and ``logs/parakh.log``.

Handler attachment is idempotent: repeated calls to :func:`get_logger` never
duplicate handlers, which matters because ingestion is re-entrant.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from src.core.constants import LOG_DATE_FORMAT, LOG_DIR, LOG_FILE_NAME, LOG_FORMAT

_ROOT_LOGGER_NAME = "parakh"
_CONFIGURED = False


def configure_logging(
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    to_file: bool = True,
) -> logging.Logger:
    """Configure the ``parakh`` logger tree exactly once.

    Args:
        level: Minimum level emitted to the stream handler.
        log_dir: Directory for the log file. Defaults to ``logs/``.
        to_file: When False, only the stream handler is attached (used by tests
            so that a read-only filesystem can never fail the suite).

    Returns:
        The configured root ``parakh`` logger.
    """
    global _CONFIGURED

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if _CONFIGURED:
        root.setLevel(level)
        return root

    root.setLevel(logging.DEBUG)
    root.propagate = False
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if to_file:
        target_dir = Path(log_dir) if log_dir is not None else LOG_DIR
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                target_dir / LOG_FILE_NAME, encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover - filesystem dependent
            root.warning("File logging disabled (%s)", exc)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child of the ``parakh`` logger.

    Args:
        name: Component name, typically ``__name__``.

    Returns:
        A logger named ``parakh.<name>``.
    """
    configure_logging()
    short = name.split(".")[-1]
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")


def reset_logging() -> None:
    """Detach all handlers. Test-only helper for isolating log assertions."""
    global _CONFIGURED
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)
    _CONFIGURED = False
