#!/usr/bin/env python
import argparse
from functools import partial
from pathlib import Path
from typing import Callable

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from mlxtk import inout, log, plot, units
from mlxtk.parameter_selection import load_scan
from mlxtk.parameters import Parameters

LOGGER = log.get_logger(__name__)
plot.make_headless()


def plot_expval(
    index: int,
    path: Path,
    parameters: Parameters,
    file_path: Path,
    extension: str = ".pdf",
    modfunc: Callable[[Figure, Axes, Parameters], None] = None,
):
    total_path = path / file_path
    try:
        fig, ax = plot.create_subplots(1, 1)
        plot.plot_expval(ax, *inout.expval.read_expval(total_path))
        ax.set_title(r"$\rho_1(x,t)$")
        system = units.get_default_unit_system()
        ax.set_xlabel(system.get_time_unit().format_label("t"))
        ax.set_xlabel(system.get_length_unit().format_label("x"))
        if modfunc:
            modfunc(fig, ax, parameters)
        plot.save(fig, str(index) + extension)
        plot.close_figure(fig)
    except FileNotFoundError:
        LOGGER.warning("file does not exist: %s", total_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scan_dir",
        type=Path,
        help="directory of the scan containing the file scan.pickle",
    )
    parser.add_argument(
        "expval", type=Path, help="relative path to expval within each simulation"
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default=".pdf",
        help="file extensions for the plots",
    )
    parser.add_argument("-o", "--output", type=str, help="name of the output directory")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    if not args.output:
        args.output = "_".join(args.expval.with_suffix("").parts)

    def apply_args(fig: Figure, ax: Axes, parameters: Parameters):
        del parameters
        plot.apply_2d_args(ax, fig, args)

    load_scan(args.scan_dir).plot_foreach(
        args.output,
        partial(
            plot_expval,
            file_path=args.expval,
            modfunc=apply_args,
            extension=args.extension,
        ),
    )


if __name__ == "__main__":
    main()
