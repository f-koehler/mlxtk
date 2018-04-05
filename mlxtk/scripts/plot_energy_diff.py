#!/usr/bin/env python
import argparse

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from ..plot.energy import plot_energy_diff


def main():
    parser = argparse.ArgumentParser(
        description=
        "Plot the difference in evolution of energy between two simulations")
    mlxtk.plot.argparser.add_plotting_arguments(parser)
    parser.add_argument(
        "--in1", type=str, dest="input_file1", help="first input_file")
    parser.add_argument(
        "--in2", type=str, dest="input_file2", help="second input_file")
    parser.add_argument("-r", "--relative", action="store_true")
    args = parser.parse_args()

    def init_plot(plot):
        plot_energy_diff(plot, args.input_file1, args.input_file2,
                         args.relative)

    program = SimplePlotProgram("Energy Diff", init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
