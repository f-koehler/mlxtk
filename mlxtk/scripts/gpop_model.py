import argparse

import matplotlib.pyplot as plt

from ..inout import read_gpop
from ..plot import create_gpop_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path",
                        nargs="?",
                        default="gpop.hdf5",
                        help="path to the gpop file")
    parser.add_argument("-d",
                        "--dof",
                        type=int,
                        default=1,
                        help="degree of freedom")
    args = parser.parse_args()

    time, grid, density = read_gpop(args.path, dof=args.dof)
    create_gpop_model(time, grid, density)

    plt.show()


if __name__ == "__main__":
    main()
