import logging
from logging import debug, info, basicConfig

# logging.basicConfig(level=logging.INFO)
basicConfig(level=logging.DEBUG)


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
