import subprocess
from matplotlib import rc as mpl_rc
from . import log


def is_lualatex_available():
    try:
        subprocess.check_output(["lualatex", "--version"])
        return True
    except FileNotFoundError:
        return False


def is_xelatex_available():
    try:
        subprocess.check_output(["xelatex", "--version"])
        return True
    except FileNotFoundError:
        return False


def is_pdflatex_available():
    try:
        subprocess.check_output(["pdflatex", "--version"])
        return True
    except FileNotFoundError:
        return False


logger = log.get_logger("mpl")

if is_lualatex_available():
    logger.info("use LuaLaTeX for matplotlib text")
    mpl_rc("text", usetex=True)
    mpl_rc("pgf", texsystem="lualatex")
elif is_xelatex_available():
    logger.info("use XeLaTeX for matplotlib text")
    mpl_rc("text", usetex=True)
    mpl_rc("pgf", texsystem="xelatex")
elif is_pdflatex_available():
    logger.info("use XeLaTeX for matplotlib text")
    mpl_rc("text", usetex=True)
    mpl_rc("pgf", texsystem="pdflatex")
