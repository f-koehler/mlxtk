import argparse

import matplotlib.pyplot as plt

from ..inout.gpop import read_gpop_hdf5
from ..plot.gpop import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?", default="gpop.hdf5", help="path to the gpop file"
    )
    parser.add_argument("-d", "--dof", type=int, default=1, help="degree of freedom")
    args = parser.parse_args()

    time, grid, density = read_gpop_hdf5(args.path, dof=args.dof)
    create_model(time, grid, density)

    plt.show()


if __name__ == "__main__":
    main()
