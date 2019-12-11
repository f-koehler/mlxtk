import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from ... import units
from ...inout import read_gpop
from ...plot import (add_argparse_2d_args, add_argparse_save_arg,
                     apply_2d_args, handle_saving, plot_gpop)
from ...tools.gpop import transform_to_momentum_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        nargs="?",
                        type=Path,
                        default=Path("propagate.h5/gpop"),
                        help="path to the gpop file")
    parser.add_argument("-d",
                        "--dof",
                        type=int,
                        default=1,
                        help="degree of freedom")
    parser.add_argument("--momentum",
                        action="store_true",
                        help="whether to transform to momentum space")
    add_argparse_2d_args(parser)
    add_argparse_save_arg(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    data = read_gpop(args.path, dof=args.dof)
    if args.momentum:
        data = transform_to_momentum_space(data)
    plot_gpop(ax, *data)

    unitsys = units.get_default_unit_system()
    try:
        ax.set_xlabel(unitsys.get_time_unit().format_label("t"))
    except units.MissingUnitError:
        ax.set_xlabel("$t$")

    try:
        ax.set_ylabel(unitsys.get_length_unit().format_label("x"))
    except units.MissingUnitError:
        ax.set_ylabel("$x$")

    apply_2d_args(ax, figure, args)

    handle_saving(figure, args)

    plt.show()


if __name__ == "__main__":
    main()