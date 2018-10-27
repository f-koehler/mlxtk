"""Handle logging

This modules provides facility to log status, warning and error messages
within mlxtk.
"""
import logging
import os

try:
    import colorama

    logging.addLevelName(
        logging.INFO,
        colorama.Fore.GREEN + logging.getLevelName(logging.INFO) +
        colorama.Style.RESET_ALL, )
    logging.addLevelName(
        logging.DEBUG,
        colorama.Fore.WHITE + logging.getLevelName(logging.DEBUG) +
        colorama.Style.RESET_ALL, )
    logging.addLevelName(
        logging.WARNING,
        colorama.Fore.YELLOW + logging.getLevelName(logging.WARNING) +
        colorama.Style.RESET_ALL, )
    logging.addLevelName(
        logging.ERROR,
        colorama.Fore.RED + logging.getLevelName(logging.ERROR) +
        colorama.Style.RESET_ALL, )
    logging.addLevelName(
        logging.CRITICAL,
        colorama.Fore.RED + logging.getLevelName(logging.ERROR) +
        colorama.Style.RESET_ALL, )
except ImportError:
    pass

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
"""str: Format for log messages."""

LOGGERS = set()
"""set: All created loggers."""

FILE_HANDLER = None
"""logging.FileHandler: Handler for the log file"""

logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)


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


def open_log_file(path: str, mode: str="a"):
    """Open the log file

    Args:
        path (str): Path of the log file
        mode (str): File mode of the log file
    """
    global FILE_HANDLER

    close_log_file()

    dirname = os.path.dirname(path)
    if dirname and (not os.path.exists(dirname)):
        os.makedirs(dirname)

    FILE_HANDLER = logging.FileHandler(path, mode)
    FILE_HANDLER.setFormatter(logging.Formatter(LOG_FORMAT))

    for logger in LOGGERS:
        logger.addHandler(FILE_HANDLER)


def close_log_file():
    """Close the log file
    """
    global FILE_HANDLER

    if FILE_HANDLER is not None:
        for logger in LOGGERS:
            logger.removeHandler(FILE_HANDLER)
        FILE_HANDLER.close()
