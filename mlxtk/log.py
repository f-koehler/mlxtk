import logging
import io
from logging import debug, info, warn, basicConfig
from logging import INFO, DEBUG, WARNING

logging.basicConfig(level=logging.INFO)

# basicConfig(level=logging.DEBUG)


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
