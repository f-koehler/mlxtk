import argparse

import matplotlib.pyplot as plt
import numpy

from ..inout.natpop import read_natpop
from ..plot.entropy import plot_entropy
from ..plot.plot import add_argparse_2d_args, apply_2d_args
from ..tools.entropy import compute_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="*",
        default=["natpop"],
        help="path to the natpop file")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    for path in args.path:
        time, natpop = read_natpop(path, node=args.node, dof=args.dof)
        entropy = compute_entropy(natpop)
        plot_entropy(ax, time, entropy)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
