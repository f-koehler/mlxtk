import argparse
from pathlib import Path

from ...inout import dmat
from ...log import get_logger
from ...tools.correlation import compute_g1, save_g1

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dmat", type=Path)
    parser.add_argument("-o", "--output", default="g1.h5", type=Path)
    args = parser.parse_args()

    LOGGER.info("read dmat")
    data_dmat = dmat.read_dmat_gridrep_hdf5(args.dmat, "dmat_gridrep")
    g1 = compute_g1(data_dmat)
    save_g1(args.output, g1)


if __name__ == "__main__":
    main()
