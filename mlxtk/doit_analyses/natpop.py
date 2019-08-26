import itertools
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy

from ..inout.natpop import read_natpop, read_natpop_hdf5
from ..parameter_selection import load_scan
from ..plot import PlotArgs2D, plot_natpop
from ..util import make_path
from .collect import collect_values
from .plot import direct_plot, doit_plot_individual


def scan_plot_natpop(scan_dir: Union[Path, str],
                     propagation: str = "propagate",
                     node: int = 1,
                     dof: int = 1,
                     extensions: List[str] = [".png", ".pdf"],
                     **kwargs):
    scan_dir = make_path(scan_dir)

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.logy = kwargs.get("logy", True)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        data = read_natpop(str(scan_dir / "by_index" / str(index) /
                               propagation / "propagate.h5") + "/natpop",
                           node=node,
                           dof=dof)
        fig, axis = plt.subplots(1, 1)
        plot_natpop(axis, *data)
        return fig, [axis]

    return doit_plot_individual(selection,
                                "natpop_{}_{}".format(node, dof),
                                [str(Path(propagation) / "propagate.h5")],
                                plot_func,
                                plotting_args,
                                extensions,
                                decorator_funcs=kwargs.get(
                                    "decorator_funcs", []))


def collect_max_depletion(scan_dir: Union[Path, str],
                          propagation_name: str = "propagate",
                          output_file: Union[Path, str] = None,
                          node: int = 1,
                          dof: int = 1,
                          missing_ok: bool = True):
    if output_file is None:
        output_file = Path("data") / "max_depletion" / (
            make_path(scan_dir).name + ".txt")

    def fetch(index, path, parameters):
        _, data = read_natpop_hdf5(path / propagation_name / "propagate.h5",
                                   "natpop",
                                   node=node,
                                   dof=dof)
        return (1 - data[:, 0]).max()

    return collect_values(scan_dir, [
        Path(propagation_name) / "propagate.h5",
    ],
                          output_file,
                          fetch,
                          missing_ok=missing_ok)


def plot_max_depletion(data_file: Union[Path, str],
                       output_dir: Union[Path, str] = None,
                       propagation_name: str = "propagate",
                       decorator_funcs=[]):
    def plot():
        fig, ax = plt.subplots(1, 1)
        data = numpy.loadtxt(data_file)

        if data.shape[1] != 2:
            raise RuntimeError("Expect only two columns in file")

        x, y = data.T

        ax.plot(x, y)

        return fig, ax

    return direct_plot(data_file,
                       output_dir,
                       plot,
                       decorator_funcs=decorator_funcs)
