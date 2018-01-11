#!/usr/bin/env python
import argparse
import numpy
import scipy.interpolate

from mlxtk import mpl
import mlxtk.plot.argparser
from mlxtk.plot.plot_program import SimplePlotProgram
from mlxtk.inout.output import read_output
from mlxtk import log


def main():
    parser = argparse.ArgumentParser(
        description=
        "Plot the difference in evolution of energy between two simulations")
    mlxtk.plot.argparser.add_plotting_arguments(parser)
    parser.add_argument(
        "--in1", type=str, dest="input_file1", help="first input_file")
    parser.add_argument(
        "--in2", type=str, dest="input_file2", help="second input_file")
    args = parser.parse_args()

    data1 = read_output(args.input_file1)
    data2 = read_output(args.input_file2)

    t_min = max(min(data1.time), min(data2.time))
    t_max = min(max(data1.time), max(data2.time))
    n_t = max(len(data1.time), len(data2.time))

    def init_plot(plot):
        plot.axes.plot(data1.time, data1.energy - data2.energy, marker=".")
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel(
            r"${\langle H\rangle}_1 (t)-{\langle H\rangle}_2 (t)$")

    def init_plot_interpolate(plot):
        t = numpy.linspace(t_min, t_max, n_t)
        interp1 = scipy.interpolate.interp1d(
            data1.time,
            data1.energy,
            kind=5,
            bounds_error=True,
            assume_sorted=True)
        interp2 = scipy.interpolate.interp1d(
            data2.time,
            data2.energy,
            kind=5,
            bounds_error=True,
            assume_sorted=True)
        plot.axes.plot(t, interp1(t) - interp2(t), marker=".")
        plot.axes.set_xlabel("$t$")
        plot.axes.set_ylabel(
            r"${\langle H\rangle}_1 (t)-{\langle H\rangle}_2 (t)$ (interpolated)")

    if not numpy.array_equal(data1.time.as_matrix(), data2.time.as_matrix()):
        log.get_logger(__name__).warn("incompatible times, interpolating")
        program = SimplePlotProgram("Energy Diff", init_plot_interpolate)
    else:
        program = SimplePlotProgram("Energy Diff", init_plot)

    program.main(args)


if __name__ == "__main__":
    main()
