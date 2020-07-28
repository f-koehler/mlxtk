import argparse
from pathlib import Path

import h5py

from mlxtk.inout import dmat, g1
from mlxtk.log import get_logger
from mlxtk.tools.correlation import compute_g1, compute_g1_diff

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dmat", type=Path)
    parser.add_argument("-o", "--output", default="g1.h5", type=Path)
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    LOGGER.info("read dmat")
    data_dmat = dmat.read_dmat_gridrep_hdf5(args.dmat, "dmat_gridrep")

    if args.diff:
        data = compute_g1_diff(data_dmat, args.normalize)
    else:
        if args.normalize:
            parser.error("The --normalize flag requires the --dif flag to be set.")
        data = compute_g1(data_dmat)

    with h5py.File(args.output, "w") as fptr:
        g1.add_g1_to_hdf5(fptr, data)


if __name__ == "__main__":
    main()
