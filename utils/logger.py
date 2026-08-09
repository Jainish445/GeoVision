"""
utils/logger.py — Unified logging setup for the entire pipeline.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: str = "results/pipeline.log",
               level: str = "INFO") -> logging.Logger:
    """Return a named logger with file + console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
