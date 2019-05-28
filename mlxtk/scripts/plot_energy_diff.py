import argparse

import matplotlib.pyplot as plt
import numpy

from .. import units
from ..inout import read_output
from ..plot import add_argparse_2d_args, apply_2d_args, plot_energy_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=2, help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time1, _, energy1, _ = read_output(args.path[0])
    time2, _, energy2, _ = read_output(args.path[1])

    if time1.shape != time2.shape:
        raise ValueError("different number of time points")

    if not numpy.allclose(time1, time2):
        raise ValueError("different time points")

    plot_energy_diff(ax, time1, energy1, energy2)

    ax.set_xlabel(units.get_time_label(working_directory=args.path[0]))
    ax.set_ylabel(
        units.get_energy_label(quantity=r"\Delta E",
                               working_directory=args.path[0]))

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
