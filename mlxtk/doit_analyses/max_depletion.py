from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy

from ..inout.natpop import read_natpop_hdf5
from ..util import make_path
from .collect import collect_values
from .plot import direct_plot


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
