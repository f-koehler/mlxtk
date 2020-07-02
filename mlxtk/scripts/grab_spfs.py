import argparse
from pathlib import Path

import h5py
import numpy
from scipy.special import binom

from mlxtk.inout.psi import read_psi_ascii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("psi", nargs="?", type=Path, default="psi")
    parser.add_argument("-o", "--output", type=Path, default="spfs.h5")
    args = parser.parse_args()

    tape, times, frames = read_psi_ascii(args.psi)
    N = tape[1]
    m = tape[3]
    n = tape[8]
    num_frames = len(times)
    num_coeffs = int(binom(N + m - 1, N))

    only_spfs = numpy.zeros(
        shape=(num_frames, frames.shape[1] - num_coeffs), dtype=frames.dtype
    )
    for i, _ in enumerate(times):
        only_spfs[i] = frames[i][num_coeffs:]

    only_spfs = numpy.reshape(only_spfs, (num_frames, m, n))

    with h5py.File(args.output, "w") as fptr:
        fptr.create_dataset("spfs", shape=only_spfs.shape, dtype=only_spfs.dtype)[
            :, :, :
        ] = only_spfs


if __name__ == "__main__":
    main()
