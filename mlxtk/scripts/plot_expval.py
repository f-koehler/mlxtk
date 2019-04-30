import argparse

import matplotlib.pyplot as plt

from ..inout import read_expval
from ..plot import add_argparse_2d_args, apply_2d_args, plot_expval
from ..util import labels_from_paths
from .. import units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="+", help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)
    labels = labels_from_paths(args.path)

    for path, label in zip(args.path, labels):
        time, values = read_expval(path)
        plot_expval(ax, time, values, label=label)

    ax.set_xlabel(units.get_time_label(working_directory=args.path[0]))

    apply_2d_args(ax, args)

    if len(args.path) > 1:
        ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
