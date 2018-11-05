import argparse

import matplotlib.pyplot as plt
import numpy

from ..inout.output import read_output
from ..plot.energy import plot_energy_diff
from ..plot.plot import add_argparse_2d_args, apply_2d_args


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

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
