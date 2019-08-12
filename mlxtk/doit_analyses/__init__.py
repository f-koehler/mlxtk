import itertools
import pickle
from pathlib import Path
from typing import Any, Callable, List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy

from .. import inout, plot
from ..inout.natpop import read_natpop_hdf5
from ..log import get_logger
from ..parameter_selection import ParameterSelection, load_scan
from ..parameters import Parameters
from ..plot import PlotArgs2D
from ..util import make_path

LOGGER = get_logger(__name__)


def doit_plot_individual(
        selection: ParameterSelection,
        plot_name: str,
        file_deps: List[Union[str, Path]],
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

    max_index_len = len(str(max(selection.parameters, key=lambda x: x[0])[0]))
    padding_format = "{:0" + str(max_index_len) + "d}"

    for (index, parameters), path in zip(selection.parameters,
                                         selection.get_paths()):

        def action_plot(index, path, parameters, targets):
            fig, axes = plot_func(index, path, parameters)

            for axis in axes:
                plotting_args.apply(axis)
                for decorator_func in decorator_funcs:
                    decorator_func(fig, axis, index, path, parameters)

            for target in targets:
                path = Path(target).parent
                if not path.exists:
                    path.mkdir(parents=True)
                plot.save(fig, target)

            plot.close_figure(fig)

        other_deps = [
            str(selection.path / "by_index" / str(index) / dep)
            for dep in file_deps
        ]

        yield {
            "name":
            "{}:{}:index_{}:plot".format(scan_name, plot_name,
                                         index).replace("=", "_"),
            "file_dep": [str(pickle_file)] + other_deps,
            "targets": [
                output_dir / (padding_format.format(index) + extension)
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
            fig, axis = plt.subplots(1, 1)
            plot.gpop.plot_gpop(axis, time, grid, density)
            return fig, [axis]

        generators.append(
            doit_plot_individual(selection,
                                 "gpop_{}".format(dof),
                                 [str(Path(propagation) / "propagate.h5")],
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
    plotting_args.logy = kwargs.get("logy", True)

    generators = []
    for scan_dir in scan_dirs:
        selection = load_scan(scan_dir)

        def plot_func(index, path, parameters):
            del path
            del parameters

            data = inout.natpop.read_natpop(
                str(scan_dir / "by_index" / str(index) / propagation /
                    "propagate.h5") + "/natpop",
                node=node,
                dof=dof)
            fig, axis = plt.subplots(1, 1)
            plot.natpop.plot_natpop(axis, *data)
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


def collect_values(scan_dir: Union[Path, str],
                   data_files: List[Union[Path, str]],
                   output_file: Union[Path, str],
                   fetch_func,
                   missing_ok: bool = True):
    scan_dir = make_path(scan_dir)
    data_files = [make_path(p) for p in data_files]
    output_file = make_path(output_file)

    selection = load_scan(scan_dir)
    file_deps = []
    for data_file in data_files:
        for i, _ in selection.parameters:
            p = scan_dir / "by_index" / str(i) / data_file
            if missing_ok and p.exists():
                file_deps.append(p)

    def action_collect_values(scan_dir: Path, targets):
        selection = load_scan(scan_dir)
        variables = selection.get_variable_names()

        def helper(index, path, parameters):
            return [parameters[variable] for variable in variables
                    ], fetch_func(index, path, parameters)

        parameters = []
        values = []
        for param, val in selection.foreach(helper, parallel=False):
            if val is not None:
                parameters.append(param)
                values.append(val)
            else:
                LOGGER.warning("cannot fetch value(s) for parameters: %s",
                               str(param))

        parameters = numpy.array(parameters, dtype=object)
        values = numpy.array(values, dtype=object)
        if len(values.shape) == 1:
            values = values.reshape((len(values), 1))
        elif len(values.shape) == 2:
            pass
        else:
            raise RuntimeError("Invalid dimensions {}".format(len(
                values.shape)))

        data = numpy.c_[parameters, values]
        header = [variable for variable in variables
                  ] + ["value{}".format(i) for i in range(values.shape[1])]
        Path(targets[0]).parent.mkdir(parents=True, exist_ok=True)
        numpy.savetxt(targets[0], data, header=" ".join(header))

    yield {
        "name":
        "{}:collect_values:{}".format(str(scan_dir.name),
                                      str(output_file.stem)).replace("=", "_"),
        "targets": [str(output_file)],
        "file_dep":
        file_deps,
        "clean":
        True,
        "actions": [(action_collect_values, [scan_dir])]
    }


def collect_max_depletion(scan_dir: Union[Path, str],
                          output_file: Union[Path, str] = None,
                          propagation_name: str = "propagate",
                          node: int = 1,
                          dof: int = 1,
                          missing_ok: bool = True):
    if output_file is None:
        output_file = Path("data") / ("max_depletion_{}.txt".format(
            make_path(scan_dir).name))

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
