import shutil
import subprocess
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.figure
import matplotlib.pyplot

from ..log import get_logger
from ..util import make_path
from .energy import plot_energy, plot_energy_diff
from .entropy import plot_entropy, plot_entropy_diff
from .expval import plot_expval
from .gpop import create_gpop_model, plot_gpop
from .natpop import plot_depletion, plot_natpop

LOGGER = get_logger(__name__)


def make_headless():
    matplotlib.use("agg")


def save(figure: matplotlib.figure.Figure,
         path: Union[str, Path],
         crop: bool = True,
         optimize: bool = True):
    path = make_path(path)

    if path.suffix == ".pdf":
        save_pdf(figure, path, crop, optimize)
        return

    if crop:
        figure.savefig(path, bbox_inches="tight")
    else:
        figure.savefig(path)


def save_pdf(figure: matplotlib.figure.Figure,
             path: Union[str, Path],
             crop: bool = True,
             optimize: bool = True):
    path = make_path(path)

    figure.savefig(path)

    if crop:
        crop_pdf(path)

    if optimize:
        optimize_pdf(path)


def crop_pdf(path: Union[str, Path]):
    path = make_path(path)

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


def optimize_pdf(path: Union[str, Path]):
    path = str(path)

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


def add_argparse_2d_args(parser):
    parser.add_argument("--logx",
                        action="store_true",
                        dest="logx",
                        help="use log scale on the x-axis")
    parser.add_argument(
        "--no-logx",
        action="store_false",
        dest="logx",
        help="do not use log scale on the x-axis",
    )
    parser.add_argument("--logy",
                        action="store_true",
                        dest="logy",
                        help="use log scale on the y-axis")
    parser.add_argument(
        "--no-logy",
        action="store_false",
        dest="logy",
        help="do not use log scale on the y-axis",
    )
    parser.add_argument("--xmin", type=float, help="minimum for the x axis")
    parser.add_argument("--xmax", type=float, help="maximum for the x axis")
    parser.add_argument("--ymin", type=float, help="minimum for the y axis")
    parser.add_argument("--ymax", type=float, help="maximum for the y axis")
    parser.add_argument("--grid",
                        action="store_true",
                        dest="grid",
                        help="draw a grid")
    parser.add_argument("--no-grid",
                        action="store_false",
                        dest="grid",
                        help="do not draw a grid")


def apply_2d_args(ax, args):
    if args.logx:
        ax.set_xscale("log")

    if args.logy:
        ax.set_yscale("log")

    xlims = list(ax.get_xlim())
    if args.xmin is not None:
        xlims[0] = args.xmin
    if args.xmax is not None:
        xlims[1] = args.xmax
    ax.set_xlim(xlims)

    ylims = list(ax.get_ylim())
    if args.ymin is not None:
        ylims[0] = args.ymin
    if args.ymax is not None:
        ylims[1] = args.ymax
    ax.set_ylim(ylims)

    ax.grid(args.grid)
