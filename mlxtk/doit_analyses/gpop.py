import itertools
from pathlib import Path
from typing import List, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from mlxtk.doit_analyses.plot import doit_plot_individual
from mlxtk.doit_analyses.video import create_slideshow
from mlxtk.inout.gpop import read_gpop
from mlxtk.log import get_logger
from mlxtk.parameter_selection import load_scan
from mlxtk.plot import PlotArgs2D
from mlxtk.plot.gpop import plot_gpop, plot_gpop_momentum
from mlxtk.util import list_files, make_path

LOGGER = get_logger(__name__)


def scan_plot_gpop(
    scan_dir: Union[Path, str],
    propagation: str = "propagate",
    dof: int = 1,
    extensions: List[str] = [".png",],
    **kwargs,
):
    scan_dir = make_path(scan_dir)

    plot_name = kwargs.get("plot_name", "gpop_{}".format(dof))

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.grid = kwargs.get("grid", False)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        time, grid, density = read_gpop(
            str(scan_dir / "by_index" / str(index) / propagation / "propagate.h5")
            + "/gpop",
            dof=dof,
        )
        fig, axis = plt.subplots(1, 1)
        plot_gpop(axis, time, grid, density)
        return fig, [axis]

    yield doit_plot_individual(
        selection,
        plot_name,
        [str(Path(propagation) / "propagate.h5")],
        plot_func,
        plotting_args,
        extensions,
        decorator_funcs=kwargs.get("decorator_funcs", []),
    )


def scan_plot_gpop_momentum(
    scan_dir: Union[Path, str],
    propagation: str = "propagate",
    dof: int = 1,
    extensions: List[str] = [".png",],
    **kwargs,
):
    scan_dir = make_path(scan_dir)

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.grid = kwargs.get("grid", False)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        time, grid, density = read_gpop(
            str(scan_dir / "by_index" / str(index) / propagation / "propagate.h5")
            + "/gpop",
            dof=dof,
        )
        fig, axis = plt.subplots(1, 1)
        plot_gpop_momentum(axis, time, grid, density)
        return fig, [axis]

    yield doit_plot_individual(
        selection,
        "gpop_momentum_{}".format(dof),
        [str(Path(propagation) / "propagate.h5")],
        plot_func,
        plotting_args,
        extensions,
        decorator_funcs=kwargs.get("decorator_funcs", []),
    )


def scan_gpop_slideshow(
    scan_dir: Union[Path, str], dof: int = 1, duration: float = 20.0
):
    scan_dir = make_path(scan_dir)
    yield create_slideshow(
        list_files(scan_dir / "plots" / ("gpop_" + str(dof)), [".png",]),
        (Path("videos") / ("gpop_" + str(dof)) / scan_dir.name).with_suffix(
            scan_dir.suffix + ".mp4"
        ),
        duration,
    )
