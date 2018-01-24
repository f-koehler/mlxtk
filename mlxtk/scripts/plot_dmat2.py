#!/usr/bin/env python
import argparse
import numpy

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from mlxtk.inout.dmat2 import read_dmat2


def main():
    parser = argparse.ArgumentParser(description="Plot dmat2")
    mlxtk.plot.argparser.add_plotting_arguments(parser)
    parser.add_argument(
        "--in",
        type=str,
        dest="input_file",
        default="output",
        help="input_file (defaults to \"output\")")
    args = parser.parse_args()

    def init_plot(plot):
        data = read_dmat2(args.input_file)

        shape = (len(numpy.unique(data["dof1"])),
                 len(numpy.unique(data["dof2"])))
        x = numpy.reshape(data["dof1"].values, shape)
        y = numpy.reshape(data["dof2"].values, shape)
        element = numpy.reshape(data["element"].values, shape)

        del data

        heatmap = plot.axes.pcolormesh(x, y, element)

        cbar = plot.figure.colorbar(heatmap)
        cbar.ax.set_ylabel("density")

    program = SimplePlotProgram("Dmat2", init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
