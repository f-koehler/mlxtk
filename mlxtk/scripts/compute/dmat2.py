import argparse
import re
import subprocess
import tempfile
from pathlib import Path

import h5py

from mlxtk.cwd import WorkingDir
from mlxtk.inout import dmat2
from mlxtk.inout.psi import read_psi_ascii, write_psi_ascii
from mlxtk.log import get_logger
from mlxtk.util import copy_file

LOGGER = get_logger(__name__)
RE_SLICE = re.compile(r"^([+-]*\d*):([+-]*\d*)(?::([+-]*\d*))?$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operator", type=Path, help="path of the operator file")
    parser.add_argument("restart", type=Path, help="path of the restart file")
    parser.add_argument("psi", type=Path, help="path of the psi file")
    parser.add_argument("--node", type=int, default=0, help="node to use")
    parser.add_argument("--dof1", type=int, default=1, help="degree of freedom to use")
    parser.add_argument(
        "--dof2", type=int, default=1, help="second degree of freedom to use"
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path.cwd() / "dmat2.h5", help="output file"
    )
    parser.add_argument("--slice", type=str)

    group_diag = parser.add_mutually_exclusive_group()
    group_diag.add_argument(
        "--diagonalize",
        action="store_true",
        help="whether to diagonalize the one-body density matrix",
    )
    group_diag.add_argument(
        "--only-diagonalize",
        action="store_true",
        help="whether to only diagonalize the one-body density matrix",
    )

    parser.add_argument(
        "--only-eigenvalues",
        action="store_true",
        help="whether to only output the eigenvalues",
    )

    group_rep = parser.add_mutually_exclusive_group()
    group_rep.add_argument(
        "--spfrep", action="store_true", help="use spf representation"
    )
    group_rep.add_argument(
        "--gridrep", action="store_true", help="use grid representation"
    )

    args = parser.parse_args()
    if args.only_eigenvalues and not (args.diagonalize or args.only_diagonalize):
        parser.error(
            "--only-eigenvalues only applies when --diagonalize or --only-diagonalize is specified"
        )

    output = args.output.resolve()
    opr = args.operator.resolve()
    rst = args.restart.resolve()
    psi = args.psi.resolve()
    with tempfile.TemporaryDirectory() as tempdir:
        with WorkingDir(tempdir):
            copy_file(opr, "opr")
            copy_file(rst, "rst")

            if args.slice:
                tape, times, psi = read_psi_ascii(psi)

                m = RE_SLICE.match(args.slice)
                if not m:
                    raise RuntimeError('Invalid slice format "{}"'.format(args.slice))

                start = 0
                end = len(times)
                step = 1
                if m.group(1) != "":
                    start = int(m.group(1))
                if m.group(2) != "":
                    stop = int(m.group(2))
                try:
                    if m.group(3) != "":
                        step = int(m.group(3))
                except IndexError:
                    pass

                indices = [i for i in range(start, end, step)]
                write_psi_ascii("psi", (tape, times[indices], psi[indices]))
            else:
                copy_file(psi, "psi")

            gridrep = args.gridrep or ((not args.gridrep) and (not args.spfrep))

            cmd = [
                "qdtk_analysis.x",
                "-opr",
                "opr",
                "-rst",
                "rst",
                "-psi",
                "psi",
                "-dmat2",
                "-nd",
                str(args.node),
                "-dof",
                str(args.dof1),
                "-dofB",
                str(args.dof2),
            ]
            if not gridrep:
                cmd.append("-spfrep")
            if gridrep:
                cmd.append("-gridrep")
            if args.diagonalize:
                cmd.append("-diagonalize")
            if args.only_diagonalize:
                cmd += ["-diagonalize", "-onlydiag"]
            if args.only_eigenvalues:
                cmd.append("-onlyeigval")
            LOGGER.info("cmd: %s", " ".join(cmd))

            subprocess.run(cmd)

            with h5py.File(output, "a") as fptr:
                if not args.only_diagonalize:
                    if gridrep:
                        dmat2.add_dmat2_gridrep_to_hdf5(
                            fptr,
                            dmat2.read_dmat2_gridrep_ascii(
                                "dmat2_dof{}_dof{}_grid".format(args.dof1, args.dof2)
                            ),
                        )
                    else:
                        dmat2.add_dmat2_spfrep_to_hdf5(
                            fptr,
                            *dmat2.read_dmat2_spfrep_ascii(
                                "dmat2_dof{}_dof{}_spf".format(args.dof1, args.dof2)
                            ),
                        )

                # if args.diagonalize or args.only_diagonalize:
                #     dmat2.add_dmat2_evals_to_hdf5(
                #         fptr,
                #         *dmat2.read_dmat2_evals_ascii(
                #             "eval_dmat2_dof{}_dof{}".format(args.dof)))
                #     if not args.only_eigenvalues:
                #         if gridrep:
                #             dmat2.add_dmat2_evecs_grid_to_hdf5(
                #                 fptr,
                #                 *dmat2.read_dmat2_evecs_grid_ascii(
                #                     "evec_dmat2_dof{}_dof{}_grid".format(
                #                         args.dof1, args.dof2)))
                #         else:
                #             dmat2.add_dmat2_evecs_spf_to_hdf5(
                #                 fptr,
                #                 *dmat2.read_dmat2_evecs_spf_ascii(
                #                     "evec_dmat2_dof{}_dof{}_spf".format(
                #                         args.dof1, args.dof2)))


if __name__ == "__main__":
    main()
