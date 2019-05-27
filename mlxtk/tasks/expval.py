"""Compute expecation values of operators.

This module hosts the implementation of a task to compute the expectation value
of an arbitary operator.

Todo:
    * Compute a static expectation value from a restart file
    * Implement case of distinguishable degrees of freedom
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from .. import cwd
from ..doit_compat import DoitAction
from ..log import get_logger
from ..util import make_path
from .task import Task


class ComputeExpectationValue(Task):
    def __init__(self, psi_or_restart: Union[str, Path],
                 operator: Union[str, Path], **kwargs):
        self.psi_or_restart = make_path(psi_or_restart)
        self.operator = make_path(operator)
        self.dirname = self.psi_or_restart.parent
        self.static = kwargs.get("static", False)

        if not self.dirname:
            self.dirname = Path(os.path.curdir)

        self.logger = get_logger(__name__ + ".ComputeExpectationValue")

        # compute the name of the expectation value based on the operator name,
        # when no name is explicitly specified
        if self.static:
            self.name = kwargs.get(
                "name",
                str(self.psi_or_restart.parent /
                    (self.psi_or_restart.name + "_" + self.operator.name)))
        else:
            self.name = kwargs.get(
                "name", str(self.psi_or_restart.parent / self.operator.name))

        # compute required paths
        self.path_operator = self.operator.with_suffix(".mb_opr")
        self.path_operator_copy = self.psi_or_restart.parent / self.operator.with_suffix(
            ".mb_opr")
        self.path_psi_or_restart = self.psi_or_restart
        self.path_expval = Path(self.name + ".exp")
        self.path_wave_function = self.dirname / "final.wfn"

        if self.static:
            self.path_psi_or_restart = self.path_psi_or_restart.with_suffix(
                ".wfn")

    def task_compute(self) -> Dict[str, Any]:
        @DoitAction
        def action_copy_operator(targets: List[str]):
            del targets

            self.logger.info("copy operator")
            shutil.copy2(self.path_operator, self.path_operator_copy)

        @DoitAction
        def action_remove_operator(targets: List[str]):
            del targets

            self.logger.info("remove operator")
            self.path_operator_copy.unlink()

        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            with cwd.WorkingDir(self.dirname):
                self.logger.info("compute expectation value")
                if self.static:
                    cmd = [
                        "qdtk_expect.x",
                        "-opr",
                        self.path_operator_copy.name,
                        "-rst",
                        self.path_psi_or_restart.name,
                        "-save",
                        self.path_expval.name,
                    ]
                else:
                    cmd = [
                        "qdtk_expect.x",
                        "-psi",
                        self.path_psi_or_restart.name,
                        "-opr",
                        self.path_operator_copy.name,
                        "-rst",
                        self.path_wave_function.name,
                        "-save",
                        self.path_expval.name,
                    ]
                self.logger.info("command: %s", " ".join(cmd))
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                subprocess.run(cmd, env=env)

        # add the copy and remove actions only when neccessary
        if self.path_operator.resolve() != self.path_operator_copy.resolve():
            return {
                "name":
                "expval:{}:compute".format(self.name),
                "actions":
                [action_copy_operator, action_compute, action_remove_operator],
                "targets": [self.path_expval],
                "file_dep": [self.path_psi_or_restart, self.path_operator],
            }

        return {
            "name": "expval:{}:compute".format(self.name),
            "actions": [action_compute],
            "targets": [self.path_expval],
            "file_dep": [self.path_psi_or_restart, self.path_operator],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_compute]
