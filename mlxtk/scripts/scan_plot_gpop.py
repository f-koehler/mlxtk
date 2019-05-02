#!/usr/bin/env python
import argparse
import os.path
from functools import partial

from .. import inout, log, plot, units
from ..parameter_selection import load_scan
from ..parameters import Parameters

LOGGER = log.get_logger(__name__)
plot.make_headless()


def plot_gpop(index: int,
              path: str,
              parameters: Parameters,
              file_path: str,
              dof: int = 1,
              extension: str = ".pdf",
              modfunc=None):
    total_path = os.path.join(path, file_path)
    try:
        fig, ax = plot.create_subplots(1, 1)
        plot.plot_gpop(ax, *inout.read_gpop(total_path, dof=1))
        ax.set_title(r"$\rho_1(x,t)$")
        ax.set_xlabel(units.get_time_label())
        ax.set_ylabel(units.get_length_label())
        if modfunc:
            modfunc(fig, ax)
        plot.save(fig, str(index) + extension)
        plot.close_figure(fig)
    except FileNotFoundError:
        LOGGER.warning("file does not exist: %s", total_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scan_dir",
        type=str,
        help="directory of the scan containing the file scan.pickle")
    parser.add_argument(
        "--file",
        type=str,
        default=os.path.join("propagate/gpop"),
        help="relative path within each simulation")
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom")
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        default=".pdf",
        help="file extensions for the plots")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    def apply_args(fig, ax):
        del fig
        plot.apply_2d_args(ax, args)

    load_scan(args.scan_dir).plot_foreach(
        "gpop",
        partial(
            plot_gpop,
            file_path=args.file,
            modfunc=apply_args,
            dof=args.dof,
            extension=args.extension))


if __name__ == "__main__":
    main()
