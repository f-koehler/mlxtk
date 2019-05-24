import argparse
import os
import shutil
import subprocess
import tempfile

import h5py
import numpy
import pandas

from ..cwd import WorkingDir
from ..inout.psi import read_first_frame
from ..log import get_logger

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "basis",
        help="wave function file containing the basis to project onto")
    parser.add_argument(
        "psi", help="wave function file containing the wave function to analyse")
    parser.add_argument(
        "-o", "--output", help="name of the output file (optional)")
    args = parser.parse_args()

    output = args.output
    if not output:
        output = "fixed_ns_{}_{}.hdf5".format(
            os.path.basename(args.psi), os.path.basename(args.basis))
    output = os.path.abspath(output)

    with tempfile.TemporaryDirectory(dir=os.getcwd()) as tmpdir:
        LOGGER.info("create a restart file from psi file")
        with open(os.path.join(tmpdir, "restart"), "w") as fp:
            fp.write(read_first_frame(args.psi))

        LOGGER.info("copy psi file")
        shutil.copy2(args.psi, os.path.join(tmpdir, "psi"))

        LOGGER.info("copy basis")
        shutil.copy2(args.basis, os.path.join(tmpdir, "basis"))

        with WorkingDir(tmpdir):
            cmd = [
                "qdtk_analysis.x", "-fixed_ns", "-rst_bra", "basis",
                "-rst_ket", "restart", "-psi", "psi", "-save", "result"
            ]
            LOGGER.info("run qdtk_analysis.x: %s", " ".join(cmd))
            subprocess.check_output(cmd)

            with open("result", "r") as fp:
                num_columns = len(fp.readline().split())

            num_coefficients = (num_columns - 1) // 3
            names = ["time"] + [
                "real_" + str(i) for i in range(num_coefficients)
            ] + ["imag_" + str(i) for i in range(num_coefficients)]
            usecols = [i for i in range(2 * num_coefficients + 1)]
            data = pandas.read_csv(
                "result",
                delim_whitespace=True,
                header=None,
                names=names,
                usecols=usecols)[names].values
            times, indices = numpy.unique(data[:, 0], return_index=True)
            num_times = len(times)

            with h5py.File("result.hdf5", "w") as fp:
                dset = fp.create_dataset(
                    "time", (num_times, ), dtype=numpy.float64)
                dset[:] = times

                dset = fp.create_dataset(
                    "real", (num_times, num_coefficients), dtype=numpy.float64)
                dset[:, :] = data[indices, 1:num_coefficients + 1]

                dset = fp.create_dataset(
                    "imag", (num_times, num_coefficients), dtype=numpy.float64)
                dset[:, :] = data[indices, num_coefficients + 1:]

            LOGGER.info("copy result")
            shutil.copy2("result.hdf5", output)


if __name__ == "__main__":
    main()
