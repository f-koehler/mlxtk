import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy

import mlxtk

RE_TIME = re.compile(r"^\s+(.+)\s+\[au\]$")
RE_ELEMENT = re.compile(r"^\s*\((.+)\,(.+)\)$")


def extract_first_frame(infile: Path, outfile: Path):
    tape_finished = False
    buffer = []

    with open(infile) as fhandle:
        for line in fhandle:
            if line.startswith("$time"):
                if tape_finished:
                    break
                tape_finished = True
            buffer.append(line)

    with open(outfile, "w") as fhandle:
        fhandle.writelines(buffer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("psi", type=Path, help="psi file to use for the recreation")
    parser.add_argument(
        "opr", type=Path, help="operator file to use for the recreation"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path.cwd(),
        help="directory for the recreated files",
    )
    args = parser.parse_args()

    path_psi = args.psi.resolve()
    path_opr = args.opr.resolve()
    path_output = args.output.resolve()

    output_hdf5 = path_output / "propagate.h5"
    output_rst = path_output / "final.wfn"
    output_psi = path_output / "psi"

    with tempfile.TemporaryDirectory() as tmpdir:
        with mlxtk.cwd.WorkingDir(tmpdir):
            mlxtk.util.copy_file(path_psi, "psi")
            mlxtk.util.copy_file(path_opr, "opr")
            extract_first_frame("psi", "rst")

            subprocess.run(
                [
                    "qdtk_analysis.x",
                    "-rst",
                    "rst",
                    "-opr",
                    "opr",
                    "-psi",
                    "psi",
                    "-recreate",
                ]
            )

            with h5py.File("result.h5", "w") as fptr:
                mlxtk.inout.gpop.add_gpop_to_hdf5(
                    fptr.create_group("gpop"), *mlxtk.inout.gpop.read_gpop_ascii("gpop")
                )
                mlxtk.inout.natpop.add_natpop_to_hdf5(
                    fptr.create_group("natpop"),
                    *mlxtk.inout.natpop.read_natpop_ascii("natpop"),
                )
                mlxtk.inout.output.add_output_to_hdf5(
                    fptr.create_group("output"),
                    *mlxtk.inout.output.read_output_ascii("output"),
                )

            shutil.move("result.h5", output_hdf5)
            shutil.move("restart", output_rst)
            shutil.move("psi", output_psi)


if __name__ == "__main__":
    main()
