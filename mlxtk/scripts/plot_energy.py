import argparse

import matplotlib.pyplot as plt

from ..inout.output import read_output
from ..plot.energy import plot_energy
from ..plot.plot import add_argparse_2d_args, apply_2d_args
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

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
