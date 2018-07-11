#!/usr/bin/env python
import argparse

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram

from ..plot.natpop import plot_natpop_diff


def main():
    parser = argparse.ArgumentParser(
        description="Plot evolution of natural population differences in time")
    mlxtk.plot.argparser.add_plotting_arguments(parser)
    parser.add_argument("input_file1", type=str)
    parser.add_argument("input_file2", type=str)
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the natural populations")
    parser.add_argument("-r", "--relative", action="store_true")
    parser.add_argument("-t", "--threshold", type=float, default=1e-3)
    args = parser.parse_args()

    def init_plot(plot):
        plot_natpop_diff(plot, args.input_file1, args.input_file2,
                         args.relative, args.threshold)

    program = SimplePlotProgram("Natural Population Differences", init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
