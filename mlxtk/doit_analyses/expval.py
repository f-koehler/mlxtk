from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt
import numpy

from mlxtk.doit_analyses.collect import collect_values
from mlxtk.doit_analyses.plot import doit_plot_individual
from mlxtk.inout.expval import read_expval_hdf5
from mlxtk.parameter_selection import load_scan
from mlxtk.plot import PlotArgs2D
from mlxtk.plot.expval import plot_expval
from mlxtk.util import make_path


def collect_initial_expval(
    scan_dir: Union[Path, str],
    expval: Union[Path, str],
    output_file: Union[Path, str] = None,
    node: int = 1,
    dof: int = 1,
    missing_ok: bool = True,
):
    expval = make_path(expval)

    if output_file is None:
        folder_name = "expval_" + expval.name.rstrip(".exp.h5")
        if not folder_name.startswith("initial_"):
            folder_name = "initial_" + folder_name
        output_file = Path("data") / (folder_name) / (make_path(scan_dir).name + ".txt")

    def fetch(index, path, parameters):
        _, data = numpy.array(read_expval_hdf5(path / expval))
        return data[0].real, data[0].imag

    return collect_values(scan_dir, [expval], output_file, fetch, missing_ok=missing_ok)


def collect_final_expval(
    scan_dir: Union[Path, str],
    expval: Union[Path, str],
    output_file: Union[Path, str] = None,
    node: int = 1,
    dof: int = 1,
    missing_ok: bool = True,
):
    expval = make_path(expval)

    if output_file is None:
        folder_name = "expval_" + expval.name.rstrip(".exp.h5")
        if not folder_name.startswith("final_"):
            folder_name = "final_" + folder_name
        output_file = Path("data") / (folder_name) / (make_path(scan_dir).name + ".txt")

    def fetch(index, path, parameters):
        _, data = numpy.array(read_expval_hdf5(path / expval))
        return data[-1].real, data[-1].imag

    return collect_values(scan_dir, [expval], output_file, fetch, missing_ok=missing_ok)


def scan_plot_expval(
    scan_dir: Union[Path, str],
    expval: Union[Path, str],
    extensions: List[str] = [".png",],
    **kwargs,
):
    scan_dir = make_path(scan_dir)
    expval = make_path(expval)

    kwargs["coefficient"] = kwargs.get("coefficient", 1.0)

    plotting_args = PlotArgs2D.from_dict(kwargs)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        data = read_expval_hdf5(
            str((scan_dir / "by_index" / str(index) / expval).with_suffix(".exp.h5"))
        )
        fig, axis = plt.subplots(1, 1)
        plot_expval(axis, *data, **kwargs)
        return fig, [axis]

    return doit_plot_individual(
        selection,
        "expval_{}".format(str(expval)).replace("/", "_"),
        [str(expval.with_suffix(".exp.h5"))],
        plot_func,
        plotting_args,
        extensions,
        decorator_funcs=kwargs.get("decorator_funcs", []),
        extra_args={"coefficient": kwargs["coefficient"]},
    )


def scan_plot_variance(
    scan_dir: Union[Path, str],
    variance: Union[Path, str],
    extensions: List[str] = [".png",],
    **kwargs,
):
    scan_dir = make_path(scan_dir)
    variance = make_path(variance)

    kwargs["coefficient"] = kwargs.get("coefficient", 1.0)

    plotting_args = PlotArgs2D.from_dict(kwargs)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        data = read_expval_hdf5(
            str((scan_dir / "by_index" / str(index) / variance).with_suffix(".var.h5"))
        )
        fig, axis = plt.subplots(1, 1)
        plot_expval(axis, *data, **kwargs)
        return fig, [axis]

    return doit_plot_individual(
        selection,
        "variance_{}".format(str(variance)).replace("/", "_"),
        [str(variance.with_suffix(".var.h5"))],
        plot_func,
        plotting_args,
        extensions,
        decorator_funcs=kwargs.get("decorator_funcs", []),
        extra_args={"coefficient": kwargs["coefficient"]},
    )
