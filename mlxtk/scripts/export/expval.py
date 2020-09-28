import argparse
from pathlib import Path

import numpy

from mlxtk.inout.expval import read_expval_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpath", type=Path)
    parser.add_argument("outputpath", type=Path)
    args = parser.parse_args()

    times, values = read_expval_hdf5(args.inputpath)
    numpy.savetxt(
        args.outputpath,
        numpy.c_[times, values.real, values.imag],
        header="times\treal\timag",
    )
