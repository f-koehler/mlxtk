"""Helpers for debugging
"""
import traceback

from .log import get_logger

LOGGER = get_logger(__name__)


def log_callstack():
    LOGGER.warning("Callstack:")
    for line in traceback.format_stack():
        LOGGER.warning("\t%s", line.strip())
