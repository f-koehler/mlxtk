import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy
import pandas

from ..cwd import WorkingDir
from ..inout.psi import read_first_frame
from ..log import get_logger
from ..tools.wave_function import load_wave_function

LOGGER = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "basis",
        type=Path,
        help="wave function file containing the basis to project onto")
    parser.add_argument(
        "psi",
        type=Path,
        help="wave function file containing the wave function to analyse")
    parser.add_argument("-o",
                        "--output",
                        type=Path,
                        help="name of the output file (optional)")
    args = parser.parse_args()

    output = args.output
    if not output:
        output = Path("fixed_ns_{}_{}.hdf5".format(args.psi.stem,
                                                   args.basis.stem))

    output = output.resolve()

    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmpdir:
        LOGGER.info("create a restart file from psi file")
        with open(Path(tmpdir) / "restart", "w") as fp:
            fp.write(read_first_frame(args.psi))

        LOGGER.info("copy psi file")
        shutil.copy2(args.psi, Path(tmpdir) / "psi")

        LOGGER.info("copy basis")
        shutil.copy2(args.basis, Path(tmpdir) / "basis")

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
            data = pandas.read_csv("result",
                                   delim_whitespace=True,
                                   header=None,
                                   names=names,
                                   usecols=usecols)[names].values
            times, indices = numpy.unique(data[:, 0], return_index=True)
            num_times = len(times)

            wfn = load_wave_function("basis")

            with h5py.File("result.hdf5", "w") as fp:
                grp = fp.create_group("fixed_ns")
                grp.attrs["N"] = wfn._tape[1]
                grp.attrs["m"] = wfn._tape[3]

                dset = grp.create_dataset("time", (num_times, ),
                                          dtype=numpy.float64)
                dset[:] = times

                dset = grp.create_dataset("real",
                                          (num_times, num_coefficients),
                                          dtype=numpy.float64)
                dset[:, :] = data[indices, 1:num_coefficients + 1]

                dset = grp.create_dataset("imag",
                                          (num_times, num_coefficients),
                                          dtype=numpy.float64)
                dset[:, :] = data[indices, num_coefficients + 1:]

            LOGGER.info("copy result")
            shutil.copy2("result.hdf5", output)


if __name__ == "__main__":
    main()
