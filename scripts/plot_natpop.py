#!/usr/bin/env python
from mlxtk.inout.natpop import read_natpop
from mlxtk.plot.natpop import plot_natpop
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot natural populations")
    parser.add_argument(
        "-o",
        nargs="?",
        default=None,
        metavar="plot_file",
        help="file name of the generated plot")
    parser.add_argument(
        "data",
        nargs="?",
        default="natpop",
        help="file containing the output of a qdtk_*.x program")
    args = parser.parse_args()

    data = read_natpop(args.data)
    if args.o:
        plot_natpop(data).save(args.o)
    else:
        plot_natpop(data).show()


if __name__ == "__main__":
    main()
