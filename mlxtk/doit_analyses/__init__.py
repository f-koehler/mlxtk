import itertools
import pickle
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt

from .. import inout, plot
from ..parameter_selection import ParameterSelection, load_scan
from ..parameters import Parameters
from ..plot import PlotArgs2D
from ..util import make_path


def doit_plot_individual(
        selection: ParameterSelection,
        plot_name: str,
        plot_func: Callable[[int, str, Parameters], Tuple[matplotlib.figure.
                                                          Figure, matplotlib.
                                                          axes.Axes]],
        plotting_args: PlotArgs2D = None,
        extensions: List[str] = [".pdf", ".png"],
        decorator_funcs: List[Callable[[
            matplotlib.figure.Figure, matplotlib.axes.
            Axes, int, str, Parameters
        ], Any]] = []):
    if plotting_args is None:
        plotting_args = PlotArgs2D()

    plot_dir = selection.path / "plots"
    output_dir = plot_dir / plot_name
    pickle_file = plot_dir / plot_name / "args.pickle"
    scan_name = selection.path.name

    def action_write_pickle(targets):
        dir = Path(targets[0]).parent
        if not dir.exists():
            dir.mkdir(parents=True)
        with open(targets[0], "wb") as fptr:
            pickle.dump([plotting_args, len(decorator_funcs)], fptr)

    yield {
        "name": "{}:{}:pickle".format(scan_name, plot_name).replace("=", "_"),
        "targets": [str(pickle_file)],
        "clean": True,
        "actions": [action_write_pickle]
    }

    for (index, parameters), path in zip(selection.parameters,
                                         selection.get_paths()):

        def action_plot(index, path, parameters, targets):
            fig, axes = plot_func(index, path, parameters)

            for ax in axes:
                plotting_args.apply(ax)
                for decorator_func in decorator_funcs:
                    decorator_func(fig, ax, index, path, parameters)

            for target in targets:
                path = Path(target).parent
                if not path.exists:
                    path.mkdir(parents=True)
                plot.save(fig, target)

            plot.close_figure(fig)

        yield {
            "name":
            "{}:{}:index_{}:plot".format(scan_name, plot_name,
                                         index).replace("=", "_"),
            "file_dep": [str(pickle_file)],
            "targets": [
                output_dir / (str(index) + extension)
                for extension in extensions
            ],
            "clean":
            True,
            "actions": [(action_plot, [index, path, parameters])],
        }


def scan_plot_gpop(scan_dirs: Union[Path, str, List[str], List[Path]],
                   propagation: str = "propagate",
                   dof: int = 1,
                   extensions: List[str] = [".png", ".pdf"],
                   **kwargs):
    if isinstance(scan_dirs, list):
        scan_dirs = [make_path(p) for p in scan_dirs]
    else:
        scan_dirs = [make_path(scan_dirs)]

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.grid = kwargs.get("grid", False)

    generators = []
    for scan_dir in scan_dirs:
        selection = load_scan(scan_dir)

        def plot_func(index, path, parameters):
            del path
            del parameters

            time, grid, density = inout.gpop.read_gpop(
                str(scan_dir / "by_index" / str(index) / propagation /
                    "propagate.h5") + "/gpop",
                dof=dof)
            fig, ax = plt.subplots(1, 1)
            plot.gpop.plot_gpop(ax, time, grid, density)
            return fig, [ax]

        generators.append(
            doit_plot_individual(selection,
                                 "gpop_{}".format(dof),
                                 plot_func,
                                 plotting_args,
                                 extensions,
                                 decorator_funcs=kwargs.get(
                                     "decorator_funcs", [])))

    for element in itertools.chain(generators):
        yield element


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
    plotting_args.grid = kwargs.get("logy", True)

    generators = []
    for scan_dir in scan_dirs:
        selection = load_scan(scan_dir)

        def plot_func(index, path, parameters):
            del index
            del path

            data = inout.natpop.read_natpop(
                str(scan_dir / "by_index" / str(index) / propagation /
                    "propagate.h5") + "/natpop",
                node=node,
                dof=dof)
            fig, ax = plt.subplots(1, 1)
            plot.natpop.plot_natpop(ax, *data)
            return fig, [ax]

        generators.append(
            doit_plot_individual(selection,
                                 "natpop_{}_{}".format(node, dof),
                                 plot_func,
                                 plotting_args,
                                 extensions,
                                 decorator_funcs=kwargs.get(
                                     "decorator_funcs", [])))

    for element in itertools.chain(generators):
        yield element
