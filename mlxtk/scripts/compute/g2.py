import argparse
from pathlib import Path

from ...inout import dmat, dmat2
from ...log import get_logger
from ...tools.correlation import compute_g2, save_g2

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dmat", type=Path)
    parser.add_argument("dmat2", type=Path)
    parser.add_argument("-o", "--output", default="g2.h5", type=Path)
    args = parser.parse_args()

    LOGGER.info("read dmat")
    data_dmat = dmat.read_dmat_gridrep_hdf5(args.dmat, "dmat_gridrep")
    LOGGER.info("read dmat2")
    data_dmat2 = dmat2.read_dmat2_gridrep_hdf5(args.dmat2, "dmat2_gridrep")
    g2 = compute_g2(data_dmat, data_dmat2)
    save_g2(args.output, g2)


if __name__ == "__main__":
    main()
