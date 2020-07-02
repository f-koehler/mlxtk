import argparse

import matplotlib.pyplot as plt
import numpy

from mlxtk import units
from mlxtk.inout import read_dmat_evals, read_natpop
from mlxtk.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path_natpop", nargs="?", default="natpop", help="path to the natpop file"
    )
    parser.add_argument(
        "path_evals",
        nargs="?",
        default="eval_dmat_dof1",
        help="path to the file containing the dmat eigenvalues",
    )
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument("-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    time, natpop = read_natpop(args.path_natpop, node=args.node, dof=args.dof)
    time2, evals = read_dmat_evals(args.path_evals)

    if not numpy.allclose(time, time2):
        raise RuntimeError("time points do not match")

    for natpop, value in zip(natpop.T, evals.T):
        plt.plot(time, natpop - value)

    system = units.get_default_unit_system()
    ax.set_xlabel(system.get_time_unit().format_label("t"))

    apply_2d_args(ax, figure, args)

    plt.show()


if __name__ == "__main__":
    main()
