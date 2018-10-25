import argparse

import matplotlib.pyplot as plt

from ..inout.expval import read_expval
from ..plot.expval import plot_expval
from ..plot.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, values = read_expval(args.path)
    plot_expval(ax, time, values)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
