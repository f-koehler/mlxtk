import argparse
from pathlib import Path

import h5py

from mlxtk.inout.gpop import read_gpop_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dof", type=int, default=1)
    parser.add_argument("datafile", type=Path)
    parser.add_argument("outputfile", type=Path)
    args = parser.parse_args()

    time, grid, densities = read_gpop_hdf5(args.datafile, "gpop", dof=args.dof)

    args.outputfile.parent.mkdir(exist_ok=True, parents=True)

    with h5py.File(args.outputfile, "w") as fptr:
        fptr.create_dataset("time", shape=time.shape, dtype=time.dtype)[:] = time
        fptr.create_dataset("grid", shape=grid.shape, dtype=grid.dtype)[:] = grid
        fptr.create_dataset("density", shape=densities.shape, dtype=densities.dtype)[
            :, :
        ] = densities


if __name__ == "__main__":
    main()
