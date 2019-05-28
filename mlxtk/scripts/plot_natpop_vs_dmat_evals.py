import argparse

import matplotlib.pyplot as plt

from .. import units
from ..inout import read_dmat_evals, read_natpop
from ..plot import add_argparse_2d_args, apply_2d_args
from ..plot.natpop import plot_natpop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path_natpop",
                        nargs="?",
                        default="natpop",
                        help="path to the natpop file")
    parser.add_argument(
        "path_evals",
        nargs="?",
        default="eval_dmat_dof1",
        help="path to the file containing the dmat eigenvalues")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument("-d",
                        "--dof",
                        type=int,
                        default=1,
                        help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, natpop = read_natpop(args.path_natpop, node=args.node, dof=args.dof)
    time2, evals = read_dmat_evals(args.path_evals)

    if not numpy.allclose(time, time2):
        raise RuntimeError("time points do not match")

    for natpop, value in zip(natpop.T, evals.T):
        plt.plot(time, natpop - value)

    ax.set_xlabel(units.get_time_label(working_directory=args.path_evals))

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
