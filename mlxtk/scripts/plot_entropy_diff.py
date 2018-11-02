import argparse

import matplotlib.pyplot as plt
import numpy

from ..inout.natpop import read_natpop
from ..plot.entropy import plot_entropy_diff
from ..plot.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs=2, default="natpop.hdf5", help="path to the natpop file")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument(
        "-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time1, natpop1 = read_natpop(args.path[0], node=args.node, dof=args.dof)
    time2, natpop2 = read_natpop(args.path[1], node=args.node, dof=args.dof)

    if time1.shape != time2.shape:
        raise RuntimeError("Number of time points differs")

    if not numpy.allclose(time1, time2):
        raise RuntimeError("Time points differ")

    entropy1 = numpy.sum(-natpop1 * numpy.log(natpop1), axis=1)
    entropy2 = numpy.sum(-natpop2 * numpy.log(natpop2), axis=1)

    plot_entropy_diff(ax, time1, entropy1, entropy2)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
