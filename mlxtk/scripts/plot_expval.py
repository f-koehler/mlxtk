#!/usr/bin/env python
import argparse

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from mlxtk.inout.expval import read_expval


def main():
    parser = argparse.ArgumentParser(
        description="Plot evolution of an expectation value over time")
    mlxtk.plot.argparser.add_plotting_arguments(parser)
    parser.add_argument("--in", type=str, dest="input_file", help="input file")
    parser.add_argument("--imaginary", action="store_true")
    parser.add_argument("--no-real", dest="real", action="store_false")
    parser.set_defaults(real=True)
    parser.set_defaults(imaginary=False)
    args = parser.parse_args()

    def init_plot(plot):
        data = read_expval(args.input_file)

        lines = []

        if args.real:
            lines += plot.axes.plot(
                data.time,
                data.real,
                color="C0",
                label=r"$\mathrm{Re}\left[\left<O\right>\right](t)$")
            plot.axes.set_ylabel(r"$\mathrm{Re}\left[\left<O\right>\right](t)$")

        if args.imaginary:
            ax = plot.axes.twinx() if args.real else plot.axes
            lines += ax.plot(
                data.time,
                data.imaginary,
                color="C1",
                label=r"$\mathrm{Im}\left[\left<O\right>\right](t)$")
            ax.set_ylabel(r"$\mathrm{Im}\left[\left<O\right>\right](t)$")

        plot.axes.set_xlabel("$t$")
        if lines:
            plot.axes.legend(lines, [l.get_label() for l in lines])

    program = SimplePlotProgram("Expectation Value", init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
