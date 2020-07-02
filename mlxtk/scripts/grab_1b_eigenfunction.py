import argparse
from pathlib import Path

import h5py

from mlxtk import get_logger
from mlxtk.inout.one_body_operator_matrix import read_one_body_operator_matrix
from mlxtk.tools.diagonalize import diagonalize_1b_operator

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="path to the one-body operator matrix")
    parser.add_argument(
        "index", type=int, help="index of the eigenfunction to write out"
    )
    parser.add_argument("output", type=Path, help="name of the output file")
    args = parser.parse_args()

    index = args.index

    _, _, matrix = read_one_body_operator_matrix(args.path)
    evals, evecs = diagonalize_1b_operator(matrix, index + 1)

    LOGGER.info("eigenvalue = %f", evals[index])
    with h5py.File(args.output, "w") as fptr:
        fptr.create_dataset(
            "wave_function", shape=evecs[index].shape, dtype=evecs[index].dtype
        )[:] = evecs[index]


if __name__ == "__main__":
    main()
