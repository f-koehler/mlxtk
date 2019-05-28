import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import units
from ..inout.natpop import read_natpop
from ..plot import add_argparse_2d_args, apply_2d_args
from ..plot.natpop import plot_natpop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        nargs="?",
                        type=Path,
                        default=Path("natpop"),
                        help="path to the natpop file")
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument("-d",
                        "--dof",
                        type=int,
                        default=1,
                        help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, natpop = read_natpop(args.path, node=args.node, dof=args.dof)
    plot_natpop(ax, time, natpop)

    ax.set_xlabel(units.get_time_label(working_directory=args.path))

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
