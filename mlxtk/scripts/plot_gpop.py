import argparse

import matplotlib.pyplot as plt

from ..inout.gpop import read_gpop
from ..plot.gpop import plot_gpop
from ..plot.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="gpop", help="path to the gpop file")
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, grid, density = read_gpop(args.path, dof=args.dof)
    plot_gpop(ax, time, grid, density)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
