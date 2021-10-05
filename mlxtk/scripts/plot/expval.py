import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import scipy.fftpack

import mlxtk
from mlxtk import units
from mlxtk.inout.expval import read_expval_hdf5
from mlxtk.plot import (
    add_argparse_2d_args,
    add_argparse_save_arg,
    apply_2d_args,
    handle_saving,
    plot_expval,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="+", help="path to the output file")
    parser.add_argument(
        "--fft",
        action="store_true",
        help="whether to transform the signal to frequency space",
    )
    parser.add_argument("--xname", help="", default="time")
    add_argparse_2d_args(parser)
    add_argparse_save_arg(parser)
    args = parser.parse_args()

    figure, ax = plt.subplots(1, 1)

    unitsys = mlxtk.units.get_default_unit_system()

    for file in args.path:
        name = file.stem
        time, values = read_expval_hdf5(file, xname=args.xname)
        if args.fft:
            plot_expval(
                ax, *mlxtk.tools.signal.fourier_transform(time, values), label=name
            )

            ax.set_xlabel((1 / unitsys.get_time_unit()).format_label(r"\omega"))
            ax.set_ylabel(
                mlxtk.units.ArbitraryUnit().format_label(r"\mathrm{amplitude}")
            )
        else:
            plot_expval(ax, time, values, label=name)
            if args.xname == "time":
                ax.set_xlabel(unitsys.get_time_unit().format_label("t"))
            else:
                ax.set_xlabel(args.xname)

    if len(args.path) > 1:
        ax.legend()

    apply_2d_args(ax, figure, args)

    handle_saving(figure, args)

    if not args.output:
        plt.show()


if __name__ == "__main__":
    main()
