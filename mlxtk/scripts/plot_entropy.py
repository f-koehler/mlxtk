import argparse

import matplotlib.pyplot as plt
import numpy

from ..inout.natpop import read_natpop
from ..plot.entropy import plot_entropy
from ..plot.plot import add_argparse_2d_args, apply_2d_args
from ..tools.entropy import compute_entropy
from ..util import labels_from_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="*", default=["natpop"], help="path to the natpop file")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    labels = labels_from_paths(args.path)
    _, ax = plt.subplots(1, 1)

    for path, label in zip(args.path, labels):
        time, natpop = read_natpop(path, node=args.node, dof=args.dof)
        entropy = compute_entropy(natpop)
        plot_entropy(ax, time, entropy, label=label)

    apply_2d_args(ax, args)
    if len(args.path) > 1:
        ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
