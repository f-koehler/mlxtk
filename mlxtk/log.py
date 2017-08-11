from logging import basicConfig
from logging import getLogger
from logging import getLevelName
from logging import addLevelName
from logging import INFO
from logging import DEBUG
from logging import WARNING
from logging import ERROR
from logging import CRITICAL

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

basicConfig(
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s", level=DEBUG)
