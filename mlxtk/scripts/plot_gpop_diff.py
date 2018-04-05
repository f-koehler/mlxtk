#!/usr/bin/env python
import argparse
import numpy
import scipy.interpolate

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from mlxtk.inout.gpop import read_gpop
from mlxtk import log

from ..plot.gpop import plot_gpop_diff


def main():
    defaults = mlxtk.plot.argparser.get_defaults()
    defaults["grid"] = False
    parser = argparse.ArgumentParser(
        "Plot the difference between two densities")
    mlxtk.plot.argparser.add_plotting_arguments(parser, defaults)
    parser.add_argument(
        "--in1", type=str, dest="input_file1", help="first input_file")
    parser.add_argument(
        "--in2", type=str, dest="input_file2", help="second input_file")
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the density")
    parser.add_argument("-r", "--relative", action="store_true")
    args = parser.parse_args()

    def init_plot(plot):
        plot_gpop_diff(plot, args.input_file1, args.input_file2, args.dof,
                       args.relative)

    program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
