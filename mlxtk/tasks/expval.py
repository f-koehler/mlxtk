import os
import shutil
import subprocess
from typing import Any, Callable, Dict, List

from .. import cwd, log
from ..doit_compat import DoitAction
from .task import Task

LOGGER = log.get_logger(__name__)


class ComputeExpectationValue(Task):
    def __init__(self, psi: str, operator: str, **kwargs):
        self.psi = psi
        self.operator = operator
        self.dirname = os.path.dirname(psi)

        # compute the name of the expectation value based on the operator name,
        # when no name is explicitly specified
        self.name = kwargs.get(
            "name",
            os.path.join(os.path.dirname(psi), os.path.basename(operator)))

        # compute required paths
        self.path_operator = self.operator + ".mb_opr"
        self.path_operator_copy = os.path.join(
            os.path.dirname(psi), self.operator + ".mb_opr")
        self.path_psi = self.psi
        self.path_expval = self.name + ".exp"
        self.path_wave_function = os.path.join(self.dirname, "final.wfn")

    def task_compute(self) -> Dict[str, Any]:
        @DoitAction
        def action_copy_operator(targets: List[str]):
            del targets

            LOGGER.info("copy operator")
            shutil.copy2(self.path_operator, self.path_operator_copy)

        @DoitAction
        def action_remove_operator(targets: List[str]):
            del targets

            LOGGER.info("remove operator")
            os.remove(self.path_operator_copy)

        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            with cwd.WorkingDir(self.dirname):
                LOGGER.info("compute expectation value")
                cmd = [
                    "qdtk_expect.x",
                    "-psi",
                    os.path.basename(self.path_psi),
                    "-opr",
                    os.path.basename(self.path_operator_copy),
                    "-rst",
                    os.path.basename(self.path_wave_function),
                    "-save",
                    os.path.basename(self.path_expval),
                ]
                LOGGER.info("command: %s", " ".join(cmd))
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                subprocess.run(cmd, env=env)

        # add the copy and remove actions only when neccessary
        if os.path.realpath(self.path_operator) != os.path.realpath(
                self.path_operator_copy):
            return {
                "name":
                "expval:{}:compute".format(self.name),
                "actions":
                [action_copy_operator, action_compute, action_remove_operator],
                "targets": [self.path_expval],
                "file_dep": [self.path_psi, self.path_operator],
            }
        else:
            return {
                "name": "expval:{}:compute".format(self.name),
                "actions": [action_compute],
                "targets": [self.path_expval],
                "file_dep": [self.path_psi, self.path_operator],
            }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_compute]
