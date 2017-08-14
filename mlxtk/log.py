from logging import CRITICAL
from logging import DEBUG
from logging import ERROR
from logging import FileHandler
from logging import Formatter
from logging import INFO
from logging import StreamHandler
from logging import WARNING
from logging import addLevelName
from logging import basicConfig
from logging import getLevelName
from logging import getLogger

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

format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

basicConfig(format=format, level=DEBUG)
formatter = Formatter(format)
