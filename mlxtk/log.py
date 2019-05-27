"""Handle logging

This modules provides facility to log status, warning and error messages
within mlxtk.
"""
import contextlib
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

try:
    import colorama

    logging.addLevelName(
        logging.INFO,
        colorama.Fore.GREEN + logging.getLevelName(logging.INFO) +
        colorama.Style.RESET_ALL,
    )
    logging.addLevelName(
        logging.DEBUG,
        colorama.Fore.WHITE + logging.getLevelName(logging.DEBUG) +
        colorama.Style.RESET_ALL,
    )
    logging.addLevelName(
        logging.WARNING,
        colorama.Fore.YELLOW + logging.getLevelName(logging.WARNING) +
        colorama.Style.RESET_ALL,
    )
    logging.addLevelName(
        logging.ERROR,
        colorama.Fore.RED + logging.getLevelName(logging.ERROR) +
        colorama.Style.RESET_ALL,
    )
    logging.addLevelName(
        logging.CRITICAL,
        colorama.Fore.RED + logging.getLevelName(logging.ERROR) +
        colorama.Style.RESET_ALL,
    )
except ImportError:
    pass

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
"""str: Format for log messages."""

LOGGERS = set()
"""set: All created loggers."""

FILE_HANDLER = None
"""logging.FileHandler: Handler for the log file"""

logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)


def get_logger(name: str) -> logging.Logger:
    """Create a logger with a given name

    If :py:data:`mlxtk.log.FILE_HANDLER` is initialized then the handler is
    added to the new logger.

    Args:
        name (str): Name for the new logger.

    Returns:
        logging.Logger: The newly created logger.
    """
    logger = logging.getLogger(name)
    if FILE_HANDLER is not None:
        logger.addHandler(FILE_HANDLER)
    LOGGERS.add(logger)
    return logger


def open_log_file(path: Path, mode: str = "a"):
    """Open the log file

    Args:
        path (str): Path of the log file
        mode (str): File mode of the log file
    """
    global FILE_HANDLER

    close_log_file()

    if path.parent and (not path.parent.exists()):
        os.makedirs(path.parent)

    FILE_HANDLER = logging.FileHandler(path, mode)
    FILE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))

    for logger in LOGGERS:
        logger.addHandler(FILE_HANDLER)


def close_log_file():
    """Close the log file
    """
    if FILE_HANDLER is not None:
        for logger in LOGGERS:
            logger.removeHandler(FILE_HANDLER)
        FILE_HANDLER.close()


class TqdmRedirectionFile:
    def __init__(self, original):
        self.original = original

    def write(self, x: str):
        if len(x.strip()) > 0:
            tqdm.write(x, file=self.original)

    def flush(self):
        return getattr(self.original, "flush", lambda: None)()


@contextlib.contextmanager
def redirect_for_tqdm():
    originals = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(TqdmRedirectionFile, originals)
        yield originals[0]
    except Exception as ex:
        raise ex
    finally:
        sys.stdout, sys.stderr = originals
