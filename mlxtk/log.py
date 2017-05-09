from logging import debug, info, warn, basicConfig
from logging import INFO, DEBUG, WARNING

import colorama
import logging
import io

logging.addLevelName(
    logging.INFO, colorama.Fore.GREEN + "INFO    " + colorama.Style.RESET_ALL)
logging.addLevelName(
    logging.DEBUG, colorama.Fore.WHITE + "DEBUG   " + colorama.Style.RESET_ALL)
logging.addLevelName(
    logging.WARNING, colorama.Fore.YELLOW +
    logging.getLevelName(logging.WARNING) + colorama.Style.RESET_ALL)
logging.addLevelName(logging.ERROR,
                     colorama.Fore.RED + "ERROR   " + colorama.Style.RESET_ALL)
logging.addLevelName(logging.CRITICAL,
                     colorama.Fore.RED + "CRITICAL" + colorama.Style.RESET_ALL)
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s  %(levelname)s  %(message)s")


def draw_box(text, min_width=80):
    lines = text.splitlines()
    max_line_length = len(max(lines, key=lambda line: len(line)))

    width = max(min_width, max_line_length + 4)
    border = ""
    for i in range(0, width):
        border += "*"

    info(border)
    for line in lines:
        line = "* " + line
        while len(line) < width - 1:
            line += " "
        line += "*"
        info(line)
    info(border)


def underline(text):
    lines = text.splitlines()
    max_line_length = len(max(lines, key=lambda line: len(line)))

    for line in lines:
        info(line)

    line = ""
    for i in range(0, max_line_length):
        line += "-"

    info(line)


class LogWrapper(io.TextIOBase):

    def __init__(self, level):
        self.level = level

    def write(s):
        if self.level == INFO:
            return info(s)

        if self.level == DEBUG:
            return info(s)

        if self.level == WARNING:
            return warn(s)
