import argparse
import subprocess
import tempfile
from pathlib import Path

import h5py

from mlxtk.cwd import WorkingDir
from mlxtk.inout.dmat2 import add_dmat2_gridrep_to_hdf5, read_dmat2_gridrep_ascii
from mlxtk.util import copy_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--restart", type=Path, default=Path("restart"), help="path to the restart file"
    )
    parser.add_argument(
        "--operator",
        type=Path,
        default=Path("operator"),
        help="path to the hamiltonian operator",
    )
    parser.add_argument(
        "--psi", type=Path, default=Path("psi"), help="path to the psi file"
    )
    parser.add_argument("--dof1", type=int, default=1, help="first degree of freedom")
    parser.add_argument("--dof2", type=int, default=1, help="second degree of freedom")
    parser.add_argument("-o", "--output", type=Path, help="path for the output file")
    args = parser.parse_args()

    restart_file = Path(args.restart).resolve()
    operator_file = Path(args.operator).resolve()
    psi_file = Path(args.psi).resolve()
    basename = "dmat2_dof{}_dof{}".format(args.dof1, args.dof2)
    if not args.output:
        output_file = Path.cwd() / (basename + "_gridrep.h5")
    else:
        output_file = Path(args.output).resolve()

    with tempfile.TemporaryDirectory() as tmpdir:
        with WorkingDir(tmpdir):
            copy_file(restart_file, "restart")
            copy_file(operator_file, "operator")
            copy_file(psi_file, "psi")
            subprocess.run(
                [
                    "qdtk_analysis.x",
                    "-dmat2",
                    "-rst",
                    "restart",
                    "-opr",
                    "operator",
                    "-psi",
                    "psi",
                    "-dof",
                    str(args.dof1),
                    "-dofB",
                    str(args.dof2),
                ]
            )
            with h5py.File(output_file, "w") as fptr:
                add_dmat2_gridrep_to_hdf5(
                    fptr, read_dmat2_gridrep_ascii(basename + "_grid")
                )


if __name__ == "__main__":
    main()
