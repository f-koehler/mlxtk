import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import h5py

from mlxtk import dvr
from mlxtk.cwd import WorkingDir
from mlxtk.doit_analyses import output
from mlxtk.doit_compat import DoitAction
from mlxtk.hashing import inaccurate_hash
from mlxtk.inout.momentum_distribution import (
    add_momentum_distribution_to_hdf5,
    read_momentum_distribution_ascii,
)
from mlxtk.tasks.task import Task
from QDTK.Operator import trafo_to_momentum_rep


class MCTDHBMomentumDistribution(Task):
    def __init__(
        self,
        psi: str,
        operator: str,
        wfn: str,
        grid: dvr.DVRSpecification,
        output_file: Optional[str] = None,
    ):
        self.psi = psi
        self.name = psi.replace("/", "_")

        self.operator = operator + ".mb_opr"

        if grid.is_fft():
            self.momentum_operator = -1j * grid.get_expdvr().get_d1()
        else:
            self.momentum_operator = -1j * grid.get_d1()

        self.wfn = wfn + ".wfn"

        if output_file is None:
            self.output_file = str(Path(psi).parent / "momentum_distribution.h5")
        else:
            self.output_file = output_file

        self.pickle_file = str(Path(self.output_file).with_suffix("")) + ".pickle"

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            obj = [
                self.psi,
                self.operator,
                inaccurate_hash(self.momentum_operator.real),
                inaccurate_hash(self.momentum_operator.imag),
                self.wfn,
                self.output_file,
            ]
            with open(targets[0], "wb") as fptr:
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": "momentum_distribution:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.pickle_file],
        }

    def task_compute(self) -> Dict[str, Any]:
        @DoitAction
        def action_compute(targets: List[str]):
            path_psi = Path(self.psi).resolve()
            path_operator = Path(self.operator).resolve()
            path_wfn = Path(self.wfn).resolve()
            path_output = Path(self.output_file).resolve()
            path_temp = path_psi.parent / ("." + self.name)

            if path_temp.exists():
                shutil.rmtree(path_temp)

            path_temp.mkdir(parents=True)
            with WorkingDir(path_temp):
                shutil.copy(path_psi, "psi")
                shutil.copy(path_operator, "oper")
                shutil.copy(path_wfn, "restart")

                trafo_to_momentum_rep([self.momentum_operator,], [1,])

                cmd = [
                    "qdtk_analysis.x",
                    "-mtrafo",
                    "trafo_mom_rep",
                    "-opr",
                    "oper",
                    "-psi",
                    "psi",
                    "-rst",
                    "restart",
                ]
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")

                result = subprocess.run(cmd, env=env)
                if result.returncode != 0:
                    raise RuntimeError("Failed to run qdtk_analysis.x")

                with h5py.File(path_output, "w") as fptr:
                    add_momentum_distribution_to_hdf5(
                        fptr, *read_momentum_distribution_ascii("mom_distr_1")
                    )

            shutil.rmtree(path_temp)

        return {
            "name": "momentum_distribution:{}:compute".format(self.name),
            "actions": [action_compute],
            "targets": [self.output_file],
            "file_dep": [self.pickle_file, self.psi, self.wfn, self.operator],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_compute]
