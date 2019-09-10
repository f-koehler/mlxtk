import itertools
from pathlib import Path
from typing import List, Union

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt

from ..inout.gpop import read_gpop
from ..parameter_selection import load_scan
from ..plot import PlotArgs2D
from ..plot.gpop import plot_gpop, plot_gpop_momentum
from ..util import make_path
from .plot import doit_plot_individual


def scan_plot_gpop(scan_dir: Union[Path, str],
                   propagation: str = "propagate",
                   dof: int = 1,
                   extensions: List[str] = [".png", ".pdf"],
                   **kwargs):
    scan_dir = make_path(scan_dir)

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.grid = kwargs.get("grid", False)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        time, grid, density = read_gpop(
            str(scan_dir / "by_index" / str(index) / propagation /
                "propagate.h5") + "/gpop",
            dof=dof)
        fig, axis = plt.subplots(1, 1)
        plot_gpop(axis, time, grid, density)
        return fig, [axis]

    yield doit_plot_individual(selection,
                               "gpop_{}".format(dof),
                               [str(Path(propagation) / "propagate.h5")],
                               plot_func,
                               plotting_args,
                               extensions,
                               decorator_funcs=kwargs.get(
                                   "decorator_funcs", []))


def scan_plot_gpop_momentum(scan_dir: Union[Path, str],
                            propagation: str = "propagate",
                            dof: int = 1,
                            extensions: List[str] = [".png", ".pdf"],
                            **kwargs):
    scan_dir = make_path(scan_dir)

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.grid = kwargs.get("grid", False)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        time, grid, density = read_gpop(
            str(scan_dir / "by_index" / str(index) / propagation /
                "propagate.h5") + "/gpop",
            dof=dof)
        fig, axis = plt.subplots(1, 1)
        plot_gpop_momentum(axis, time, grid, density)
        return fig, [axis]

    yield doit_plot_individual(selection,
                               "gpop_momentum_{}".format(dof),
                               [str(Path(propagation) / "propagate.h5")],
                               plot_func,
                               plotting_args,
                               extensions,
                               decorator_funcs=kwargs.get(
                                   "decorator_funcs", []))
