import shutil
import subprocess

import matplotlib.figure

from ..log import get_logger

LOGGER = get_logger(__name__)


def save_pdf(figure: matplotlib.figure.Figure, path: str, crop: bool = True):
    figure.savefig(path)

    if not crop:
        return

    if not shutil.which("pdfcrop"):
        LOGGER.error("Cannot find pdfcrop, skip cropping")

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
