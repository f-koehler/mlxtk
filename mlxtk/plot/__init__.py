import shutil
import subprocess

import matplotlib
import matplotlib.figure
import matplotlib.pyplot

from ..log import get_logger

from .energy import plot_energy, plot_energy_diff
from .entropy import plot_entropy, plot_entropy_diff
from .expval import plot_expval
from .gpop import plot_gpop, create_gpop_model
from .natpop import plot_natpop

LOGGER = get_logger(__name__)


def make_headless():
    matplotlib.use("agg")


def save_pdf(figure: matplotlib.figure.Figure,
             path: str,
             crop: bool = True,
             optimize: bool = True):
    figure.savefig(path)

    if crop:
        crop_pdf(path)

    if optimize:
        optimize_pdf(path)


def crop_pdf(path: str):
    if not shutil.which("pdfcrop"):
        LOGGER.error("Cannot find pdfcrop, skip cropping")
        return

    gs = shutil.which("gs")
    if gs:
        try:
            output = subprocess.check_output(["gs",
                                              "--version"]).decode().strip()
            if output == "9.27":
                LOGGER.error(
                    "There is a bug when using pdfcrop with ghostscript==9.27 that might remove actual content of the PDF, skip cropping"
                )
                return
        except subprocess.CalledProcessError:
            LOGGER.warning("Failed to check Ghostscript version")

    LOGGER.info("Cropping PDF file: %s", path)
    subprocess.check_output(["pdfcrop", "--hires", path, path])

def optimize_pdf(path: str):
    if not shutil.which("pdftocairo"):
        LOGGER.error("Cannot find pdftocairo, skip PDF optimization")
        return

    LOGGER.info("Optimizing PDF file: %s", path)
    tmp_path = path + "_tmp"
    subprocess.check_output(["pdftocairo", "-pdf", path, tmp_path])
    shutil.move(tmp_path, path)


def close_figure(figure: matplotlib.figure.Figure):
    matplotlib.pyplot.close(figure)


def create_subplots(*args, **kwargs):
    return matplotlib.pyplot.subplots(*args, **kwargs)
