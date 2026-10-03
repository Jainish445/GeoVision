"""
utils/logger.py — Unified logging setup for the entire pipeline.
"""

import logging
import sys
from pathlib import Path

# Anchored to the project root so the log lands in one place regardless of cwd.
DEFAULT_LOG_FILE = Path(__file__).resolve().parent.parent / "results" / "pipeline.log"


def _force_utf8_console() -> None:
    """Windows consoles default to cp1252 and choke on '→', '═', '✔' etc."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def get_logger(name: str, log_file=DEFAULT_LOG_FILE,
               level: str = "INFO") -> logging.Logger:
    """Return a named logger with file + console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    _force_utf8_console()

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
