import argparse

import matplotlib.pyplot as plt

from ..inout.output import read_output_hdf5
from ..plot.energy import plot_energy
from ..plot.plot import add_argparse_2d_args, apply_2d_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="output.hdf5", help="path to the output file"
    )
    add_argparse_2d_args(parser)
    args = parser.parse_args()

    fig, ax = plt.subplots(1, 1)

    time, _, energy, _ = read_output_hdf5(args.path)
    plot_energy(ax, time, energy)

    apply_2d_args(ax, args)

    plt.show()


if __name__ == "__main__":
    main()
