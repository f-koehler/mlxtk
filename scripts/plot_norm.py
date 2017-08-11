#!/usr/bin/env python
from mlxtk.inout.output import read_output
from mlxtk.plot.norm import plot_norm
import argparse


def main():
    parser = argparse.ArgumentParser(description="Plot norm")
    parser.add_argument(
        "-o",
        nargs="?",
        default=None,
        metavar="plot_file",
        help="file name of the generated plot")
    parser.add_argument(
        "--abs",
        action="store_true",
        help="whether to use abs(deviation) or not")
    parser.add_argument(
        "data",
        nargs="?",
        default="output",
        help="file containing the output of a qdtk_*.x program")
    args = parser.parse_args()

    data = read_output(args.data)
    if args.o:
        plot_norm(data, absolute=args.abs).save(args.o)
    else:
        plot_norm(data, absolute=args.abs).show()


if __name__ == "__main__":
    main()
