# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
import logging
import os

try:
    import colorama
    logging.addLevelName(
        logging.INFO, colorama.Fore.GREEN + logging.getLevelName(logging.INFO)
        + colorama.Style.RESET_ALL)
    logging.addLevelName(
        logging.DEBUG, colorama.Fore.WHITE +
        logging.getLevelName(logging.DEBUG) + colorama.Style.RESET_ALL)
    logging.addLevelName(
        logging.WARNING, colorama.Fore.YELLOW +
        logging.getLevelName(logging.WARNING) + colorama.Style.RESET_ALL)
    logging.addLevelName(
        logging.ERROR, colorama.Fore.RED + logging.getLevelName(logging.ERROR)
        + colorama.Style.RESET_ALL)
    logging.addLevelName(
        logging.CRITICAL, colorama.Fore.RED +
        logging.getLevelName(logging.ERROR) + colorama.Style.RESET_ALL)
except ImportError:
    pass

LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

loggers = set()
file_handler = None
stream_handler = logging.StreamHandler()

logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)


def get_logger(name):
    logger = logging.getLogger(name)
    if file_handler is not None:
        logger.addHandler(file_handler)
    loggers.add(logger)
    return logger


def open_log_file(path, mode="a"):
    global file_handler

    close_log_file()

    dirname = os.path.dirname(path)
    if dirname and (not os.path.exists(dirname)):
        os.makedirs(dirname)

    file_handler = logging.FileHandler(path, mode)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))

    for logger in loggers:
        logger.addHandler(file_handler)


def close_log_file():
    global file_handler

    if file_handler is not None:
        for logger in loggers:
            logger.removeHandler(file_handler)
        file_handler.close()
