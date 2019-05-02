#!/usr/bin/env python
import argparse
from functools import partial
import os.path

from .. import plot
from .. import inout
from .. import log
from .. import units
from ..parameters import Parameters
from ..parameter_selection import load_scan

LOGGER = log.get_logger(__name__)
plot.make_headless()


def plot_gpop(index: int,
              path: str,
              parameters: Parameters,
              file_path: str,
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
        plot.save_pdf(fig, str(index) + ".pdf")
        plot.close_figure(fig)
    except FileNotFoundError:
        LOGGER.warning("file does not exist: %s", total_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scan_dir",
        help="directory of the scan containing the file scan.pickle")
    parser.add_argument(
        "--file",
        default=os.path.join("propagate/gpop"),
        help="relative path within each simulation")
    plot.add_argparse_2d_args(parser)
    args = parser.parse_args()

    def apply_args(fig, ax):
        del fig
        plot.apply_2d_args(ax, args)

    load_scan(args.scan_dir).plot_foreach(
        "gpop", partial(plot_gpop, file_path=args.file, modfunc=apply_args))


if __name__ == "__main__":
    main()
