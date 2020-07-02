import pickle
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import matplotlib

from mlxtk import plot
from mlxtk.parameter_selection import ParameterSelection
from mlxtk.parameters import Parameters
from mlxtk.util import make_path


def doit_plot_individual(
    selection: ParameterSelection,
    plot_name: str,
    file_deps: List[Union[str, Path]],
    plot_func: Callable[
        [int, str, Parameters], Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    ],
    plotting_args: plot.PlotArgs2D = None,
    extensions: List[str] = [".pdf", ".png"],
    decorator_funcs: List[
        Callable[
            [matplotlib.figure.Figure, matplotlib.axes.Axes, int, str, Parameters], Any
        ]
    ] = [],
    **extra_args,
):
    if plotting_args is None:
        plotting_args = plot.PlotArgs2D()

    plot_dir = selection.path / "plots"
    output_dir = plot_dir / plot_name
    pickle_file = plot_dir / plot_name / "args.pickle"
    scan_name = selection.path.name

    def action_write_pickle(targets):
        dir = Path(targets[0]).parent
        if not dir.exists():
            dir.mkdir(parents=True)
        with open(targets[0], "wb") as fptr:
            pickle.dump(
                [plotting_args, len(decorator_funcs), extra_args], fptr, protocol=3
            )

    yield {
        "name": "{}:{}:pickle".format(scan_name, plot_name).replace("=", "_"),
        "targets": [str(pickle_file)],
        "clean": True,
        "actions": [action_write_pickle],
    }

    max_index_len = len(str(max(selection.parameters, key=lambda x: x[0])[0]))
    padding_format = "{:0" + str(max_index_len) + "d}"

    for (index, parameters), path in zip(selection.parameters, selection.get_paths()):

        def action_plot(index, path, parameters, targets):
            fig, axes = plot_func(index, path, parameters)

            for axis in axes:
                plotting_args.apply(axis, fig)
                for decorator_func in decorator_funcs:
                    decorator_func(fig, axis, index, path, parameters)

            for target in targets:
                path = Path(target).parent
                if not path.exists:
                    path.mkdir(parents=True)
                plot.save(fig, target)

            plot.close_figure(fig)

        other_deps = [
            str(selection.path / "by_index" / str(index) / dep) for dep in file_deps
        ]

        yield {
            "name": "{}:{}:index_{}:plot".format(scan_name, plot_name, index).replace(
                "=", "_"
            ),
            "file_dep": [str(pickle_file)] + other_deps,
            "targets": [
                output_dir / (padding_format.format(index) + extension)
                for extension in extensions
            ],
            "clean": True,
            "actions": [(action_plot, [index, path, parameters])],
        }


def direct_plot(
    input_files: List[Union[str, Path]],
    output_file_base: Union[str, Path],
    plot_func: Callable[[], Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]],
    plotting_args: plot.PlotArgs2D = None,
    extensions: List[str] = [".pdf", ".png"],
    decorator_funcs: List[
        Callable[[matplotlib.figure.Figure, matplotlib.axes.Axes], Any]
    ] = [],
):
    if plotting_args is None:
        plotting_args = plot.PlotArgs2D()

    input_files = [make_path(input_file) for input_file in input_files]
    output_file_base = make_path(output_file_base)
    pickle_file = output_file_base.parent / ("." + output_file_base.stem + ".pickle")

    def action_write_pickle(targets):
        dirname = Path(targets[0]).parent
        if not dirname.exists():
            dirname.mkdir(parents=True)
        with open(targets[0], "wb") as fptr:
            pickle.dump([plotting_args, len(decorator_funcs)], fptr, protocol=3)

    yield {
        "name": "direct_plot:{}:pickle".format(output_file_base.name).replace("=", "_"),
        "targets": [str(pickle_file)],
        "clean": True,
        "actions": [action_write_pickle],
    }

    def action_plot(targets):
        fig, axes = plot_func()

        for axis in axes:
            plotting_args.apply(axis)
            for decorator_func in decorator_funcs:
                decorator_func(fig, axis)

        for target in targets:
            plot.save(fig, target)

        plot.close_figure(fig)

    yield {
        "name": "direct_plot:{}:plot".format(output_file_base.name).replace("=", "_"),
        "file_dep": [str(input_file) for input_file in input_files]
        + [str(pickle_file)],
        "targets": [str(output_file_base) + ext for ext in extensions],
        "clean": True,
        "actions": [action_plot,],
    }
