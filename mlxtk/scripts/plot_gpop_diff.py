#!/usr/bin/env python
import argparse
import numpy

import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from mlxtk.inout.gpop import read_gpop


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
    args = parser.parse_args()

    def init_plot(plot):
        times1, grids1, densities1 = read_gpop(args.input_file1)
        times2, grids2, densities2 = read_gpop(args.input_file2)

        if numpy.any(times1 != times2):
            raise RuntimeError("Incompatible time points")

        if numpy.any(grids1[args.dof] != grids2[args.dof]):
            raise RuntimeError("Incompatible grids")

        difference = numpy.transpose(
            densities1[args.dof] - densities2[args.dof])
        amplitude = max(abs(numpy.min(difference)), numpy.max(difference))
        t, x = numpy.meshgrid(times1, grids1[args.dof])
        heatmap = plot.axes.pcolormesh(
            t, x, difference, cmap="bwr", vmin=-amplitude, vmax=amplitude)
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel("$x$")
        cbar = plot.figure.colorbar(heatmap)
        cbar.ax.set_ylabel("density1 - density2")

    program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
