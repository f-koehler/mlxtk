import gzip
import os
import subprocess
from typing import List

from .. import cwd, log
from ..doit_compat import DoitAction
from ..inout.expval import read_expval_ascii, write_expval_hdf5

LOGGER = log.get_logger(__name__)


def compute_expectation_value(psi: str, operator: str, **kwargs):
    name = kwargs.get("name",
                      os.path.join(
                          os.path.dirname(psi), os.path.basename(operator)))

    dirname = os.path.dirname(psi)

    def task_compute():
        path_opr = operator + ".mb_opr.gz"
        path_opr_tmp = os.path.join(dirname, operator + ".mb_opr")
        path_wfn = os.path.join(dirname, "final.wfn.gz")
        path_wfn_tmp = os.path.join(dirname, "final.wfn")

        path_psi = psi
        path_psi_hdf5 = psi + ".hdf5"

        path_expval = name + ".exp"
        path_expval_hdf5 = path_expval + ".hdf5"

        @DoitAction
        def action_copy_operator(targets: List[str]):
            del targets

            LOGGER.info("copy operator")
            with gzip.open(path_opr, "rb") as fp_in:
                with open(path_opr_tmp, "w") as fp_out:
                    fp_out.write(fp_in.read().decode())

        @DoitAction
        def action_copy_wave_function(targets: List[str]):
            del targets

            LOGGER.info("copy wave function")
            with gzip.open(path_wfn, "rb") as fp_in:
                with open(path_wfn_tmp, "w") as fp_out:
                    fp_out.write(fp_in.read().decode())

        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            with cwd.WorkingDir(dirname):
                LOGGER.info("compute expectation value")
                cmd = [
                    "qdtk_expect.x",
                    "-psi",
                    os.path.basename(path_psi),
                    "-opr",
                    os.path.basename(path_opr_tmp),
                    "-rst",
                    os.path.basename(path_wfn_tmp),
                    "-save",
                    os.path.basename(path_expval),
                ]
                LOGGER.info("command: %s", " ".join(cmd))
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                subprocess.run(cmd, env=env)

        @DoitAction
        def action_convert(targets: List[str]):
            del targets

            LOGGER.info("convert expectation value to HDF5")
            write_expval_hdf5(path_expval_hdf5, read_expval_ascii(path_expval))
            os.remove(path_expval)

        @DoitAction
        def action_remove_wave_function(targets: List[str]):
            del targets

            LOGGER.info("remove wave function")
            os.remove(path_wfn_tmp)

        @DoitAction
        def action_remove_operator(targets: List[str]):
            del targets

            LOGGER.info("remove operator")
            os.remove(path_opr_tmp)

        return {
            "name":
            "expval:{}:compute".format(name),
            "actions": [
                action_copy_operator,
                action_copy_wave_function,
                action_compute,
                action_convert,
                action_remove_wave_function,
                action_remove_operator,
            ],
            "targets": [path_expval_hdf5],
            "file_dep": [path_psi_hdf5, path_opr],
        }

    return [task_compute]
