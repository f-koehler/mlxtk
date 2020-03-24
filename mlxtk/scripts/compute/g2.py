import argparse
from pathlib import Path

import h5py

from mlxtk.inout import dmat, dmat2, g2
from mlxtk.log import get_logger
from mlxtk.tools.correlation import compute_g2, compute_g2_diff

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dmat", type=Path)
    parser.add_argument("dmat2", type=Path)
    parser.add_argument("-o", "--output", default="g2.h5", type=Path)
    parser.add_argument("--diff", action="store_true")
    args = parser.parse_args()

    LOGGER.info("read dmat")
    data_dmat = dmat.read_dmat_gridrep_hdf5(args.dmat, "dmat_gridrep")
    LOGGER.info("read dmat2")
    data_dmat2 = dmat2.read_dmat2_gridrep_hdf5(args.dmat2, "dmat2_gridrep")

    if args.diff:
        data = compute_g2_diff(data_dmat, data_dmat2)
    else:
        data = compute_g2(data_dmat, data_dmat2)

    with h5py.File(args.output, "w") as fptr:
        g2.add_g2_to_hdf5(fptr, data)


if __name__ == "__main__":
    main()
