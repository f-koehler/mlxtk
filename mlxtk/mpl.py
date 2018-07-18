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

mpl_rc("figure", figsize=(4.5, 4.5))
mpl_rc("lines", markersize=2.5, linewidth=1)
mpl_rc("axes", labelsize=10)
mpl_rc("xtick", labelsize=10)
mpl_rc("ytick", labelsize=10)
mpl_rc("legend", fontsize=10)
mpl_rc("axes", titlesize=10)

latex_preamble = [
        r"\usepackage{siunitx}", r"\usepackage{bigints}", r"\usepackage{xfrac}",
        r"\usepackage{xparse}", r"\ExplSyntaxOn",
        r"\DeclareDocumentCommand\diff{o}{\IfNoValueTF{#1}{\mathrm{d}}{\mathrm{d}^{#1}}}",
        r"\ExplSyntaxOff"
    ]
mpl_rc(
    "text.latex",
    preamble=latex_preamble)
mpl_rc("pgf", preamble=latex_preamble, rcfonts=False)

if is_lualatex_available():
    logger.info("use LuaLaTeX for matplotlib text")
    mpl_rc("text", usetex=True)
    mpl_rc("pgf", texsystem="lualatex")
    usetex = True
elif is_xelatex_available():
    logger.info("use XeLaTeX for matplotlib text")
    mpl_rc("text", usetex=True)
    mpl_rc("pgf", texsystem="xelatex")
    usetex = True
elif is_pdflatex_available():
    logger.info("use XeLaTeX for matplotlib text")
    mpl_rc("text", usetex=True)
    mpl_rc("pgf", texsystem="pdflatex")
    usetex = True
else:
    usetex = False


def require_latex():
    if not usetex:
        raise RuntimeError(
            "LaTeX is required for this plot, but no compiler was found")
