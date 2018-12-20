import argparse

import matplotlib.pyplot as plt

from ..inout.natpop import read_natpop
from ..plot.natpop import plot_natpop
from ..plot.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="natpop", help="path to the natpop file")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, natpop = read_natpop(args.path, node=args.node, dof=args.dof)
    plot_natpop(ax, time, natpop)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
