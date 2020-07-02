import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy

from mlxtk import units
from mlxtk.inout.momentum_distribution import read_momentum_distribution_hdf5
from mlxtk.plot import (
    add_argparse_2d_args,
    add_argparse_save_arg,
    apply_2d_args,
    handle_saving,
    plot_gpop,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("momentum_distribution.h5"),
        help="path to the momentum_distribution file",
    )
    parser.add_argument("--colorbar", type=bool)
    add_argparse_2d_args(parser)
    add_argparse_save_arg(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    times, momenta, density = read_momentum_distribution_hdf5(
        args.path, "momentum_distribution"
    )

    average_momentum = numpy.zeros_like(times)
    for i, _ in enumerate(average_momentum):
        average_momentum[i] = numpy.sum(momenta * density[i])

    Y, X = numpy.meshgrid(momenta, times)
    mesh = ax.pcolormesh(X, Y, density)
    plt.plot(times, average_momentum)

    apply_2d_args(ax, figure, args)

    handle_saving(figure, args)

    plt.show()


if __name__ == "__main__":
    main()
