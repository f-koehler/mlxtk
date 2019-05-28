import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import units
from ..inout import read_gpop
from ..plot import add_argparse_2d_args, apply_2d_args, plot_gpop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        nargs="?",
                        type=Path,
                        default=Path("gpop"),
                        help="path to the gpop file")
    parser.add_argument("-d",
                        "--dof",
                        type=int,
                        default=1,
                        help="degree of freedom")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, grid, density = read_gpop(args.path, dof=args.dof)
    plot_gpop(ax, time, grid, density)

    ax.set_xlabel(units.get_time_label(working_directory=args.path))
    ax.set_ylabel(units.get_length_label(working_directory=args.path))

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
