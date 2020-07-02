import argparse

import matplotlib.pyplot as plt
import numpy

from mlxtk import units
from mlxtk.inout import read_natpop
from mlxtk.plot import add_argparse_2d_args, apply_2d_args, plot_entropy_diff
from mlxtk.tools.entropy import compute_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs=2, help="path to the natpop file")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument("-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    time1, natpop1 = read_natpop(args.path[0], node=args.node, dof=args.dof)
    time2, natpop2 = read_natpop(args.path[1], node=args.node, dof=args.dof)

    if time1.shape != time2.shape:
        raise ValueError("number of time points differs")

    if not numpy.allclose(time1, time2):
        raise ValueError("time points differ")

    entropy1 = compute_entropy(natpop1)
    entropy2 = compute_entropy(natpop2)

    plot_entropy_diff(ax, time1, entropy1, entropy2)

    system = units.get_default_unit_system()
    ax.set_xlabel(system.get_time_unit().format_label("t"))

    apply_2d_args(ax, figure, args)

    plt.show()


if __name__ == "__main__":
    main()
