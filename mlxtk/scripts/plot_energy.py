#!/usr/bin/env python
import sys

from mlxtk.plot.plot_program import SimplePlotProgram, create_argparser
from mlxtk.inout.output import read_output


def main():
    parser = create_argparser(
        "Plot evolution of energy expectation value over time")
    parser.add_argument(
        "--in",
        type=str,
        dest="input_file",
        default="output",
        help="input_file (defaults to \"output\")")
    args = parser.parse_args()

    def init_plot(plot):
        data = read_output(args.input_file)
        plot.axes.plot(data.time, data.energy, marker=".")
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel(r"$\langle H\rangle (t)$")

    program = SimplePlotProgram("Energy", init_plot)
    program.main(args)


if __name__ == "__main__":
    main()