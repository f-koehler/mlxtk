import itertools
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import h5py
import matplotlib.pyplot as plt
import numpy

from mlxtk.doit_analyses.collect import collect_values
from mlxtk.doit_analyses.plot import direct_plot, doit_plot_individual
from mlxtk.doit_analyses.video import create_slideshow
from mlxtk.inout.natpop import read_natpop, read_natpop_hdf5
from mlxtk.parameter_selection import load_scan
from mlxtk.plot import PlotArgs2D, plot_entropy, plot_natpop
from mlxtk.tools.entropy import compute_entropy
from mlxtk.util import list_files, make_path


def scan_plot_natpop(
    scan_dir: Union[Path, str],
    propagation: str = "propagate",
    node: int = 1,
    dof: int = 1,
    extensions: List[str] = [".png",],
    **kwargs,
):
    scan_dir = make_path(scan_dir)

    plotting_args = PlotArgs2D.from_dict(kwargs)
    plotting_args.logy = kwargs.get("logy", True)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        data = read_natpop(
            str(scan_dir / "by_index" / str(index) / propagation / "propagate.h5")
            + "/natpop",
            node=node,
            dof=dof,
        )
        fig, axis = plt.subplots(1, 1)
        plot_natpop(axis, *data)
        return fig, [axis]

    return doit_plot_individual(
        selection,
        "natpop_{}_{}".format(node, dof),
        [str(Path(propagation) / "propagate.h5")],
        plot_func,
        plotting_args,
        extensions,
        decorator_funcs=kwargs.get("decorator_funcs", []),
    )


def scan_plot_entropy(
    scan_dir: Union[Path, str],
    propagation: str = "propagate",
    node: int = 1,
    dof: int = 1,
    extensions: List[str] = [".png",],
    **kwargs,
):
    scan_dir = make_path(scan_dir)

    plotting_args = PlotArgs2D.from_dict(kwargs)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        data = read_natpop(
            str(scan_dir / "by_index" / str(index) / propagation / "propagate.h5")
            + "/natpop",
            node=node,
            dof=dof,
        )
        entropy = compute_entropy(data[1])
        fig, axis = plt.subplots(1, 1)
        plot_entropy(axis, data[0], entropy)
        return fig, [axis]

    return doit_plot_individual(
        selection,
        "entropy",
        [str(Path(propagation) / "propagate.h5")],
        plot_func,
        plotting_args,
        extensions,
        decorator_funcs=kwargs.get("decorator_funcs", []),
    )


class DefaultNatpopAnalysis:
    def __init__(
        self,
        scan_dir: Union[Path, str],
        propagation: str = "propagate",
        node: int = 1,
        dof: int = 1,
        missing_ok: bool = True,
        output_file: Union[Path, str] = None,
    ):
        self.scan_dir = make_path(scan_dir)
        self.propagation = propagation
        self.node = node
        self.dof = dof
        self.missing_ok = missing_ok

        if output_file is None:
            self.output_file = (
                Path("data")
                / "natpop_{}_{}".format(node, dof)
                / (self.scan_dir.name.replace("=", "_") + ".h5")
            )
        else:
            self.output_file = make_path(output_file)

    def __call__(self) -> Dict[str, Any]:
        pickle_obj = [self.node, self.dof, self.missing_ok]
        pickle_path = self.output_file.with_suffix(".pickle")

        pickle_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pickle_path, "wb") as fptr:
            pickle.dump(pickle_obj, fptr, protocol=3)

        def action(scan_dir, input_files, targets):
            variables, values = load_scan(scan_dir).get_variable_values()

            with h5py.File(targets[0], "w") as fptr:
                max_depletion = []
                max_entropy = []
                max_last_orbital = []

                for input_file in input_files:
                    time, data = read_natpop_hdf5(
                        input_file, "natpop", node=self.node, dof=self.dof
                    )
                    del time

                    entropy = compute_entropy(data)

                    max_depletion.append((1 - data[:, 0]).max())
                    max_entropy.append(entropy.max())
                    max_last_orbital.append(data[:, -1].max())

                dset = fptr.create_dataset(
                    "max_depletion", (len(max_depletion),), dtype=numpy.float64
                )
                dset[:] = max_depletion

                dset = fptr.create_dataset(
                    "max_entropy", (len(max_entropy),), dtype=numpy.float64
                )
                dset[:] = max_entropy

                dset = fptr.create_dataset(
                    "max_last_orbital", (len(max_last_orbital),), dtype=numpy.float64
                )
                dset[:] = max_last_orbital

                grp = fptr.create_group("variables")
                for var in variables:
                    dset = grp.create_dataset(
                        var, values[var].shape, dtype=values[var].dtype
                    )
                    dset[:] = values[var]

        scan_name_sanitized = self.scan_dir.name.replace("=", "_")

        scan = load_scan(self.scan_dir)
        input_files = [
            self.scan_dir / "by_index" / str(i) / self.propagation / "propagate.h5"
            for i, _ in scan.parameters
        ]

        yield {
            "name": "{}:natpop:{}_{}:{}:default_analysis".format(
                self.propagation, self.node, self.dof, scan_name_sanitized
            ),
            "file_dep": [pickle_path] + input_files,
            "targets": [self.output_file],
            "actions": [(action, [self.scan_dir, input_files])],
        }


def collect_max_depletion(
    scan_dir: Union[Path, str],
    propagation_name: str = "propagate",
    output_file: Union[Path, str] = None,
    node: int = 1,
    dof: int = 1,
    missing_ok: bool = True,
):
    if output_file is None:
        output_file = (
            Path("data") / "max_depletion" / (make_path(scan_dir).name + ".txt")
        )

    def fetch(index, path, parameters):
        _, data = read_natpop_hdf5(
            path / propagation_name / "propagate.h5", "natpop", node=node, dof=dof
        )
        return (1 - data[:, 0]).max()

    return collect_values(
        scan_dir,
        [Path(propagation_name) / "propagate.h5",],
        output_file,
        fetch,
        missing_ok=missing_ok,
    )


def plot_max_depletion(
    data_file: Union[Path, str],
    output_dir: Union[Path, str] = None,
    propagation_name: str = "propagate",
    decorator_funcs=[],
):
    def plot():
        fig, ax = plt.subplots(1, 1)
        data = numpy.loadtxt(data_file)

        if data.shape[1] != 2:
            raise RuntimeError("Expect only two columns in file")

        x, y = data.T

        ax.plot(x, y)

        return fig, ax

    return direct_plot(data_file, output_dir, plot, decorator_funcs=decorator_funcs)


def scan_natpop_slideshow(
    scan_dir: Union[Path, str], node: int = 1, dof: int = 1, duration: float = 20.0
):
    scan_dir = make_path(scan_dir)
    yield create_slideshow(
        list_files(scan_dir / "plots" / "natpop_{}_{}".format(node, dof), [".png",]),
        (
            Path("videos") / ("natpop_{}_{}".format(node, dof)) / scan_dir.name
        ).with_suffix(scan_dir.suffix + ".mp4"),
        duration,
    )
