import argparse

import matplotlib.pyplot as plt

from ..inout.output import read_output
from ..plot.energy import plot_energy
from ..plot.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="output",
        help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, _, energy, _ = read_output(args.path)
    plot_energy(ax, time, energy)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
