#!/usr/bin/env python
import argparse

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram

from ..plot.natpop import plot_natpop


def main():
    parser = argparse.ArgumentParser(
        description="Plot evolution of natural populations in time")
    mlxtk.plot.argparser.add_plotting_arguments(parser)
    parser.add_argument(
        "--in",
        type=str,
        dest="input_file",
        default="natpop",
        help="input_file (defaults to \"natpop\")",
    )
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the natural populations",
    )
    args = parser.parse_args()

    def init_plot(plot):
        plot_natpop(plot, args.input_file)
        plot.figure.set_tight_layout(True)

    program = SimplePlotProgram("Natural Populations", init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
