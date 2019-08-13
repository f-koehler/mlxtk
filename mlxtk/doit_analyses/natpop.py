import itertools
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt

from ..inout.natpop import read_natpop
from ..parameter_selection import load_scan
from ..plot import PlotArgs2D, plot_natpop
from ..util import make_path
from .plot import doit_plot_individual


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
