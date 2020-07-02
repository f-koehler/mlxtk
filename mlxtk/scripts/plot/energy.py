import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mlxtk import units
from mlxtk.inout import read_output
from mlxtk.plot import (
    add_argparse_2d_args,
    add_argparse_save_arg,
    apply_2d_args,
    handle_saving,
    plot_energy,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default="propagate.h5/output",
        help="path to the output file",
    )
    add_argparse_2d_args(parser)
    add_argparse_save_arg(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    time, _, energy, _ = read_output(args.path)
    plot_energy(ax, time, energy)

    system = units.get_default_unit_system()
    ax.set_xlabel(system.get_time_unit().format_label("t"))
    ax.set_ylabel(system.get_length_unit().format_label("x"))

    apply_2d_args(ax, figure, args)

    handle_saving(figure, args)
    plt.show()


if __name__ == "__main__":
    main()
