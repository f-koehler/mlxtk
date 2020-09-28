import argparse
from pathlib import Path

import numpy

from mlxtk.inout.output import read_output_hdf5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputpath", type=Path)
    parser.add_argument("outputpath", type=Path)
    args = parser.parse_args()

    time, norm, energy, overlap = read_output_hdf5(args.inputpath, "/output")
    numpy.savetxt(
        args.outputpath,
        numpy.c_[time, norm, energy, overlap],
        header="time\tnorm\tenergy\toverlap",
    )
