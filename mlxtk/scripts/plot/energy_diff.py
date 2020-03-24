import argparse

import matplotlib.pyplot as plt
import numpy

from mlxtk import units
from mlxtk.inout import read_output
from mlxtk.plot import add_argparse_2d_args, apply_2d_args, plot_energy_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=2, help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    time1, _, energy1, _ = read_output(args.path[0])
    time2, _, energy2, _ = read_output(args.path[1])

    if time1.shape != time2.shape:
        raise ValueError("different number of time points")

    if not numpy.allclose(time1, time2):
        raise ValueError("different time points")

    plot_energy_diff(ax, time1, energy1, energy2)

    system = units.get_default_unit_system()
    ax.set_xlabel(system.get_time_unit().format_label("t"))
    ax.set_ylabel(system.get_energy_unit().format_label(r"\Delta E"))

    apply_2d_args(ax, figure, args)

    plt.show()


if __name__ == "__main__":
    main()
