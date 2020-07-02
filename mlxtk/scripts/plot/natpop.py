import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mlxtk import units
from mlxtk.inout.natpop import read_natpop
from mlxtk.plot import (
    add_argparse_2d_args,
    add_argparse_save_arg,
    apply_2d_args,
    handle_saving,
)
from mlxtk.plot.natpop import plot_natpop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("propagate.h5/natpop"),
        help="path to the natpop file",
    )
    parser.add_argument("-n", "--node", type=int, default=1, help="node")
    parser.add_argument("-d", "--dof", type=int, default=1, help="degree of freedom")
    add_argparse_2d_args(parser)
    add_argparse_save_arg(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    time, natpop = read_natpop(args.path, node=args.node, dof=args.dof)
    plot_natpop(ax, time, natpop)

    try:
        ax.set_xlabel(units.get_default_unit_system().get_time_unit().format_label("t"))
    except units.MissingUnitError:
        ax.set_xlabel("$t$")

    apply_2d_args(ax, figure, args)
    handle_saving(figure, args)

    plt.show()


if __name__ == "__main__":
    main()
