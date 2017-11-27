#!/usr/bin/env python
from matplotlib.cm import get_cmap
from matplotlib.colors import LightSource
import numpy

from mlxtk.plot.plot_program import SimplePlotProgram, create_argparser
from mlxtk.inout.gpop import read_gpop


def main():
    parser = create_argparser(
        "Plot the evolution of the density of a degree of freedom")
    parser.add_argument(
        "--in",
        type=str,
        dest="input_file",
        default="gpop",
        help="input_file (defaults to \"gpop\")")
    parser.add_argument(
        "--dof",
        type=int,
        default=1,
        help="degree of freedom for which to plot the density")
    args = parser.parse_args()

    def init_plot(plot):
        time, grid, density = read_gpop(args.input_file)
        density = density[args.dof]
        grid = grid[args.dof]

        time_min_index = 0
        if args.xmin is not None:
            time_min_index = numpy.nonzero(time >= args.xmin)[0][0]

        time_max_index = len(time)
        if args.xmax is not None:
            time_max_index = numpy.nonzero(time <= args.xmax)[-1][-1] + 1

        grid_min_index = 0
        if args.ymin is not None:
            grid_min_index = numpy.nonzero(grid >= args.ymin)[0][0]

        grid_max_index = len(grid)
        if args.ymax is not None:
            grid_max_index = numpy.nonzero(grid <= args.ymax)[-1][-1] + 1

        time = time[time_min_index:time_max_index]
        grid = grid[grid_min_index:grid_max_index]
        density = density[time_min_index:time_max_index, grid_min_index:
                          grid_max_index]
        grid, time = numpy.meshgrid(grid, time)

        ls = LightSource(azdeg=0, altdeg=80)
        facecolors = ls.shade(density, get_cmap("gnuplot"))
        plot.axes.plot_surface(
            time,
            grid,
            density,
            facecolors=facecolors,
            rstride=1,
            cstride=1,
            linewidth=0,
            antialiased=False)
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel("$x$")
        plot.axes.set_zlabel("density")

    program = SimplePlotProgram(
        "Density of DOF {}".format(args.dof), init_plot, projection="3d")
    program.main(args)


if __name__ == "__main__":
    main()
