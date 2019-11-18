import argparse
from pathlib import Path

import h5py

from ...inout import dmat, g1
from ...log import get_logger
from ...tools.correlation import compute_g1

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dmat", type=Path)
    parser.add_argument("-o", "--output", default="g1.h5", type=Path)
    args = parser.parse_args()

    LOGGER.info("read dmat")
    data_dmat = dmat.read_dmat_gridrep_hdf5(args.dmat, "dmat_gridrep")
    data = compute_g1(data_dmat)

    with h5py.File(args.output, "w") as fptr:
        g1.add_g1_to_hdf5(fptr, data)


if __name__ == "__main__":
    main()
