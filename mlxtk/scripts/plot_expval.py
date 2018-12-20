import argparse

import matplotlib.pyplot as plt

from ..inout.expval import read_expval
from ..plot.expval import plot_expval
from ..plot.plot import add_argparse_2d_args, apply_2d_args
from ..util import labels_from_paths


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

    apply_2d_args(ax, args)

    if len(args.path) > 1:
        ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
