import argparse
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
import matplotlib.figure
import matplotlib.pyplot
from matplotlib.backend_bases import LocationEvent
from matplotlib.figure import Figure

from mlxtk.log import get_logger
from mlxtk.plot import cursor
from mlxtk.plot.energy import plot_energy, plot_energy_diff
from mlxtk.plot.entropy import plot_entropy, plot_entropy_diff
from mlxtk.plot.expval import plot_expval
from mlxtk.plot.gpop import create_gpop_model, plot_gpop
from mlxtk.plot.natpop import plot_depletion, plot_natpop
from mlxtk.util import make_path

LOGGER = get_logger(__name__)


def make_headless():
    matplotlib.use("agg")


def save(
    figure: matplotlib.figure.Figure,
    path: Union[str, Path],
    crop: bool = False,
    optimize: bool = False,
):
    path = make_path(path)

    if path.suffix == ".pdf":
        save_pdf(figure, path, crop, optimize)
        return

    if crop:
        figure.savefig(path, bbox_inches="tight")
    else:
        figure.savefig(path)


def save_pdf(
    figure: matplotlib.figure.Figure,
    path: Union[str, Path],
    crop: bool = True,
    optimize: bool = True,
):
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
            output = subprocess.check_output(["gs", "--version"]).decode().strip()
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


class PlotArgs2D:
    def __init__(self):
        self.xmin: Optional[float] = None
        self.xmax: Optional[float] = None
        self.ymin: Optional[float] = None
        self.ymax: Optional[float] = None
        self.logx: bool = False
        self.logy: bool = False
        self.grid: bool = True
        self.dpi: int = 600
        self.extensions: List[str] = [".pdf", ".png"]

    @staticmethod
    def from_dict(kwargs):
        args = PlotArgs2D()
        args.xmin = kwargs.get("xmin", None)
        args.xmax = kwargs.get("xmax", None)
        args.ymin = kwargs.get("ymin", None)
        args.ymax = kwargs.get("ymax", None)
        args.logx = kwargs.get("logx", False)
        args.logy = kwargs.get("logy", False)
        args.grid = kwargs.get("grid", True)
        args.dpi = kwargs.get("dpi", 600)
        return args

    @staticmethod
    def from_namespace(namespace: argparse.Namespace):
        return PlotArgs2D.from_dict(namespace.__dict__)

    def apply(self, axes: matplotlib.axes.Axes, figure: matplotlib.figure.Figure):

        axes.grid(self.grid)
        if self.logx:
            axes.set_xscale("log")
        if self.logy:
            axes.set_yscale("log")

        xmin, xmax = axes.get_xlim()
        ymin, ymax = axes.get_ylim()
        xmin = xmin if self.xmin is None else self.xmin
        xmax = xmax if self.xmax is None else self.xmax
        ymin = ymin if self.ymin is None else self.ymin
        ymax = ymax if self.ymax is None else self.ymax
        axes.set_xlim(xmin=xmin, xmax=xmax)
        axes.set_ylim(ymin=ymin, ymax=ymax)

        if self.dpi and (figure is not None):
            figure.set_dpi(self.dpi)


def add_argparse_2d_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--logx", action="store_true", dest="logx", help="use log scale on the x-axis"
    )
    parser.add_argument(
        "--no-logx",
        action="store_false",
        dest="logx",
        help="do not use log scale on the x-axis",
    )
    parser.add_argument(
        "--logy", action="store_true", dest="logy", help="use log scale on the y-axis"
    )
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
    parser.add_argument("--grid", action="store_true", dest="grid", help="draw a grid")
    parser.add_argument(
        "--no-grid", action="store_false", dest="grid", help="do not draw a grid"
    )
    parser.add_argument(
        "--dpi", type=int, default=600, help="resolution (dpi) of figure"
    )


def apply_2d_args(
    ax: matplotlib.axes.Axes,
    figure: matplotlib.figure.Figure,
    namespace: argparse.Namespace,
):
    PlotArgs2D.from_namespace(namespace).apply(ax, figure)


def add_argparse_save_arg(parser: argparse.ArgumentParser):
    parser.add_argument("-o", "--output", type=Path, help="path to the output file")
    parser.add_argument(
        "--crop-pdf", action="store_true", dest="crop_pdf", help="crop PDF file"
    )
    parser.add_argument(
        "--no-crop-pdf",
        action="store_false",
        dest="crop_pdf",
        help="do not crop PDF file",
    )
    parser.add_argument(
        "--optimize-pdf",
        action="store_true",
        dest="optimize_pdf",
        help="optimize PDF file",
    )
    parser.add_argument(
        "--no-optimize-pdf",
        action="store_false",
        dest="optimize_pdf",
        help="do not optimize PDF file",
    )
    parser.set_defaults(crop_pdf=False, optimize_pdf=False)


def handle_saving(figure: Figure, namespace: argparse.Namespace):
    if namespace.output:
        save(
            figure,
            namespace.output,
            crop=namespace.crop_pdf,
            optimize=namespace.optimize_pdf,
        )
