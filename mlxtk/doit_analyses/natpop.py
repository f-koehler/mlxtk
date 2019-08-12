import itertools
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt

from ..inout.natpop import read_natpop
from ..parameter_selection import load_scan
from ..plot import PlotArgs2D, plot_natpop
from ..util import make_path
from .plot import doit_plot_individual


def scan_plot_natpop(scan_dirs: Union[Path, str, List[str], List[Path]],
                     propagation: str = "propagate",
                     node: int = 1,
                     dof: int = 1,
                     extensions: List[str] = [".png", ".pdf"],
                     **kwargs):
    if isinstance(scan_dirs, list):
        scan_dirs = [make_path(p) for p in scan_dirs]
    else:
        scan_dirs = [make_path(scan_dirs)]

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.logy = kwargs.get("logy", True)

    generators = []
    for scan_dir in scan_dirs:
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

        generators.append(
            doit_plot_individual(selection,
                                 "natpop_{}_{}".format(node, dof),
                                 [str(Path(propagation) / "propagate.h5")],
                                 plot_func,
                                 plotting_args,
                                 extensions,
                                 decorator_funcs=kwargs.get(
                                     "decorator_funcs", [])))

    for element in itertools.chain(generators):
        yield element
