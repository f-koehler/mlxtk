# pylint: disable=wildcard-import
# pylint: disable=unused-wildcard-import
from logging import *

try:
    import colorama
    addLevelName(
        INFO,
        colorama.Fore.GREEN + getLevelName(INFO) + colorama.Style.RESET_ALL)
    addLevelName(
        DEBUG,
        colorama.Fore.WHITE + getLevelName(DEBUG) + colorama.Style.RESET_ALL)
    addLevelName(WARNING, colorama.Fore.YELLOW + getLevelName(WARNING) +
                 colorama.Style.RESET_ALL)
    addLevelName(
        ERROR,
        colorama.Fore.RED + getLevelName(ERROR) + colorama.Style.RESET_ALL)
    addLevelName(
        CRITICAL,
        colorama.Fore.RED + getLevelName(ERROR) + colorama.Style.RESET_ALL)
except ImportError:
    pass

logging_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

basicConfig(format=logging_format, level=DEBUG)
formatter = Formatter(logging_format)
