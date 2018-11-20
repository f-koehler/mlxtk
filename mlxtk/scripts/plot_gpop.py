#!/usr/bin/env python
import argparse

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from mlxtk.inout.gpop import read_gpop

from ..plot.gpop import plot_gpop


def main():
    defaults = mlxtk.plot.argparser.get_defaults()
    defaults["grid"] = False
    parser = argparse.ArgumentParser(
        "Plot the evolution of the density of a degree of freedom")
    mlxtk.plot.argparser.add_plotting_arguments(parser, defaults)
    parser.add_argument(
        "--in",
        type=str,
        dest="input_file",
        default="gpop",
        help="input_file (defaults to \"gpop\")",
    )
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the density",
    )
    parser.add_argument("--smooth", action="store_true", default=False)
    args = parser.parse_args()

    def init_plot(plot):
        plot_gpop(plot, args.input_file, args.dof, smooth=args.smooth)

    program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
