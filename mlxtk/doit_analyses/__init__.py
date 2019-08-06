import pickle
from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt

from .. import inout, plot
from ..parameter_selection import ParameterSelection, load_scan
from ..plot import PlotArgs2D
from ..util import make_path


def doit_plot_individual(selection: ParameterSelection,
                         plot_name: str,
                         plot_func,
                         plotting_args: PlotArgs2D = None,
                         extensions: List[str] = [".pdf", ".png"],
                         extra_args={}):
    if plotting_args is None:
        plotting_args = PlotArgs2D()

    plot_dir = selection.path / "plots"
    output_dir = plot_dir / plot_name
    pickle_file = plot_dir / plot_name / "args.pickle"
    scan_name = selection.path.name

    def action_write_pickle(targets):
        with open(targets[0], "wb") as fptr:
            pickle.dump([plotting_args, extra_args], fptr)

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

            for target in targets:
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

    generators = []
    for scan_dir in scan_dirs:
        selection = load_scan(scan_dir)

        def plot_func(index, path, parameters):
            time, grid, density = inout.gpop.read_gpop(
                str(scan_dir / "by_index" / str(index) / propagation /
                    "propagate.h5") + "/gpop",
                dof=dof)
            fig, ax = plt.subplots(1, 1)
            plot.gpop.plot_gpop(ax, time, grid, density)
            return fig, [ax]

        generators.append(
            doit_plot_individual(selection, "gpop_{}".format(dof), plot_func))

    for generator in generators:
        for element in generator:
            yield element
