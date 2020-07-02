from pathlib import Path
from typing import List, Union

import matplotlib.pyplot as plt

from mlxtk.doit_analyses.collect import collect_values
from mlxtk.doit_analyses.plot import doit_plot_individual
from mlxtk.inout.output import read_output_hdf5
from mlxtk.parameter_selection import load_scan
from mlxtk.plot import PlotArgs2D
from mlxtk.plot.energy import plot_energy
from mlxtk.util import make_path


def collect_final_energy(
    scan_dir: Union[Path, str],
    propagation_name: str = "propagate",
    output_file: Union[Path, str] = None,
    missing_ok: bool = True,
):
    if output_file is None:
        output_file = (
            Path("data")
            / "{}_final_energy".format(propagation_name)
            / (make_path(scan_dir).name + ".txt")
        )

    def fetch(index, path, parameters):
        _, _, energy, _ = read_output_hdf5(
            path / propagation_name / "propagate.h5", "output"
        )
        return energy[-1]

    return collect_values(
        scan_dir,
        [Path(propagation_name) / "propagate.h5",],
        output_file,
        fetch,
        missing_ok=missing_ok,
    )


def scan_plot_energy(
    scan_dir: Union[Path, str],
    propagation: str = "propagate",
    extensions: List[str] = [".png"],
    **kwargs,
):
    scan_dir = make_path(scan_dir)

    plotting_args = PlotArgs2D.from_dict(kwargs)

    selection = load_scan(scan_dir)

    def plot_func(index, path, parameters):
        del path
        del parameters

        time, _, energy, _ = read_output_hdf5(
            scan_dir / "by_index" / str(index) / propagation / "propagate.h5", "output"
        )

        fig, axis = plt.subplots(1, 1)
        plot_energy(axis, time, energy)
        return fig, [axis]

    yield doit_plot_individual(
        selection,
        "energy",
        [str(Path(propagation) / "propagate.h5")],
        plot_func,
        plotting_args,
        extensions,
        decorator_funcs=kwargs.get("decorator_funcs", []),
    )
