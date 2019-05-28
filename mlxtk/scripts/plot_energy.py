import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .. import units
from ..inout import read_output
from ..plot import add_argparse_2d_args, apply_2d_args, plot_energy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        nargs="?",
                        type=Path,
                        default=Path("output"),
                        help="path to the output file")
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    _, ax = plt.subplots(1, 1)

    time, _, energy, _ = read_output(args.path)
    plot_energy(ax, time, energy)

    ax.set_xlabel(units.get_time_label(working_directory=args.path))
    ax.set_ylabel(units.get_energy_label(working_directory=args.path))

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
