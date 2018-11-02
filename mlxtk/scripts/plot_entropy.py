import argparse
import numpy

import matplotlib.pyplot as plt

from ..inout.natpop import read_natpop
from ..plot.entropy import plot_entropy
from ..plot.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="natpop.hdf5",
        help="path to the natpop file")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, natpop = read_natpop(args.path, node=args.node, dof=args.dof)
    entropy = numpy.sum(-natpop * numpy.log(natpop), axis=1)
    plot_entropy(ax, time, entropy)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
