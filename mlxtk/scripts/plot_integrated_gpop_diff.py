#!/usr/bin/env python
import argparse

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram

from ..plot.gpop import plot_integrated_gpop_diff


def main():
    defaults = mlxtk.plot.argparser.get_defaults()
    defaults["grid"] = False
    parser = argparse.ArgumentParser(
        "Plot the difference between two densities")
    mlxtk.plot.argparser.add_plotting_arguments(parser, defaults)
    parser.add_argument("input_file1", type=str, help="first input_file")
    parser.add_argument("input_file2", type=str, help="second input_file")
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the density")
    parser.add_argument("-r", "--relative", action="store_true", default=False)
    parser.add_argument("--threshold", type=float, default=1e-5)
    args = parser.parse_args()

    def init_plot(plot):
        plot_integrated_gpop_diff(
            plot,
            args.input_file1,
            args.input_file2,
            args.dof,
            threshold=args.threshold,
            relative=args.relative)

    program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
