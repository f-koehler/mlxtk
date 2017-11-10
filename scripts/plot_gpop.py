#!/usr/bin/env python
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
        times, grids, densities = read_gpop(args.input_file)
        density = numpy.transpose(densities[1])
        t, x = numpy.meshgrid(times, grids[1])
        plot.axes.pcolormesh(t, x, density, cmap="gnuplot")
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel("$x$")

    program = SimplePlotProgram("Density of DOF {}".format(args.dof),
                                init_plot)
    program.main(args)


if __name__ == "__main__":
    main()
