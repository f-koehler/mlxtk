import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import units
from ..inout.expval import read_expval
from ..plot import add_argparse_2d_args, apply_2d_args, plot_expval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        type=Path,
                        nargs="?",
                        help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, values = read_expval(args.path)
    plot_expval(ax, time, values)

    system = units.get_default_unit_system()
    ax.set_xlabel(system.get_time_unit().format_label("t"))

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
