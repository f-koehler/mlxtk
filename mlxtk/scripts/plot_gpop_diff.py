#!/usr/bin/env python
import argparse
import numpy
import scipy.interpolate

import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from mlxtk.inout.gpop import read_gpop
from mlxtk import log


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

    times1, grids1, densities1 = read_gpop(args.input_file1)
    times2, grids2, densities2 = read_gpop(args.input_file2)

    grid1 = grids1[args.dof]
    grid2 = grids2[args.dof]

    def init_plot(plot):
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

    def init_plot_interpolate(plot):
        t_min = max(numpy.min(times1), numpy.min(times2))
        t_max = min(numpy.max(times1), numpy.max(times2))
        x_min = max(numpy.min(grid1), numpy.min(grid2))
        x_max = min(numpy.max(grid1), numpy.max(grid2))
        n_t = max(len(times1), len(times2))
        n_x = max(len(grid1), len(grid2))

        gpop1 = numpy.transpose(densities1[args.dof])
        gpop2 = numpy.transpose(densities2[args.dof])

        interp1 = scipy.interpolate.interp2d(
            times1,
            grid1,
            gpop1,
            kind="quintic",
            copy=False,
            bounds_error=True)
        interp2 = scipy.interpolate.interp2d(
            times2,
            grid2,
            gpop2,
            kind="quintic",
            copy=False,
            bounds_error=True)

        t = numpy.linspace(t_min, t_max, n_t)
        x = numpy.linspace(x_min, x_max, n_x)
        difference = interp1(t, x) - interp2(t, x)
        amplitude = max(abs(numpy.min(difference)), numpy.max(difference))

        heatmap = plot.axes.pcolormesh(
            t, x, difference, cmap="bwr", vmin=-amplitude, vmax=amplitude)
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel("$x$")
        cbar = plot.figure.colorbar(heatmap)
        cbar.ax.set_ylabel("density1 - density2 (interpolated)")

    if numpy.any(times1 != times2):
        log.warn("incompatible time points, interpolating")
        program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                    init_plot_interpolate)
    elif numpy.any(grids1[args.dof] != grids2[args.dof]):
        log.warn("incompatible grid points, interpolating")
        program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                    init_plot_interpolate)
    else:
        program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                    init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
