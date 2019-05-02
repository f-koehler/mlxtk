import argparse

import matplotlib.pyplot as plt

from .. import units
from ..inout import read_output
from ..plot import add_argparse_2d_args, apply_2d_args, plot_energy
from ..util import labels_from_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="+", default=["output"], help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    labels = labels_from_paths(args.path)

    _, ax = plt.subplots(1, 1)

    for path, label in zip(args.path, labels):
        time, _, energy, _ = read_output(path)
        plot_energy(ax, time, energy, label=label)

    if len(args.path) > 1:
        ax.legend()

    ax.set_xlabel(units.get_time_label(working_directory=args.path[0]))
    ax.set_ylabel(units.get_energy_label(working_directory=args.path[0]))

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
