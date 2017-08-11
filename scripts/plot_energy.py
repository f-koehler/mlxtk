#!/usr/bin/env python
from mlxtk.inout.output import read_output
from mlxtk.plot.energy import plot_energy
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot energy")
    parser.add_argument(
        "-o",
        nargs="?",
        default=None,
        metavar="plot_file",
        help="file name of the generated plot")
    parser.add_argument(
        "--logx",
        action="store_true",
        help="whether to use logarithmic time axis")
    parser.add_argument(
        "--logy",
        action="store_true",
        help="whether to use logarithmic energy axis")
    parser.add_argument(
        "data",
        nargs="?",
        default="output",
        help="file containing the output of a qdtk_*.x program")
    args = parser.parse_args()

    data = read_output(args.data)
    if args.o:
        plot_energy(data, logx=args.logx, logy=args.logy).save(args.o)
    else:
        plot_energy(data, logx=args.logx, logy=args.logy).show()


if __name__ == "__main__":
    main()
