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
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from .. import cwd
from ..doit_compat import DoitAction
from ..log import get_logger
from ..util import copy_file, make_path
from .task import Task


class ComputeExpectationValue(Task):
    def __init__(self, psi: Union[str, Path], operator: Union[str, Path],
                 **kwargs):
        self.logger = get_logger(__name__ + ".ComputeExpectationValue")
        self.unique_name = kwargs.get("unique_name", False)

        # compute required paths
        self.operator = make_path(operator).with_suffix(".mb_opr")
        self.psi = make_path(psi)
        if self.unique_name:
            self.expval = self.psi.with_name(
                self.psi.stem + "_" + self.operator.stem).with_suffix(".exp")
        else:
            self.expval = (self.psi.parent /
                           self.operator.stem).with_suffix(".exp")

        self.wave_function = make_path(
            kwargs.get("wave_function",
                       self.psi.parent / "final")).with_suffix(".wfn")

        self.name = str(self.expval.with_suffix(""))

    def task_compute(self) -> Dict[str, Any]:
        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            operator = self.operator.resolve()
            psi = self.psi.resolve()
            expval = self.expval.resolve()
            wave_function = self.wave_function.resolve()

            with tempfile.TemporaryDirectory() as tmpdir:
                with cwd.WorkingDir(tmpdir):
                    self.logger.info("compute expectation value")
                    copy_file(operator, "operator")
                    copy_file(psi, "psi")
                    copy_file(wave_function, "restart")
                    cmd = [
                        "qdtk_expect.x", "-opr", "operator", "-rst", "restart",
                        "-psi", "psi", "-save", "expval"
                    ]
                    self.logger.info("command: %s", " ".join(cmd))
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                    subprocess.run(cmd, env=env)
                    copy_file("expval", expval)

        return {
            "name": "expval:{}:compute".format(self.name),
            "actions": [action_compute],
            "targets": [str(self.expval)],
            "file_dep": [str(self.psi), str(self.operator)],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_compute]


class ComputeExpectationValueStatic(Task):
    def __init__(self, wave_function: Union[str, Path],
                 operator: Union[str, Path]):
        self.logger = get_logger(__name__ + ".ComputeExpectationValueStatic")

        # compute required paths
        self.operator = make_path(operator).with_suffix(".mb_opr")
        self.wave_function = make_path(wave_function).with_suffix(".wfn")
        self.expval = self.wave_function.with_name(
            self.wave_function.stem + "_" +
            self.operator.stem).with_suffix(".exp")

        self.name = str(self.expval.with_suffix(""))

    def task_compute(self) -> Dict[str, Any]:
        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            operator = self.operator.resolve()
            expval = self.expval.resolve()
            wave_function = self.wave_function.resolve()

            with tempfile.TemporaryDirectory() as tmpdir:
                with cwd.WorkingDir(tmpdir):
                    self.logger.info("compute expectation value (static)")
                    copy_file(operator, "operator")
                    copy_file(wave_function, "restart")
                    cmd = [
                        "qdtk_expect.x", "-opr", "operator", "-rst", "restart",
                        "-save", "expval"
                    ]
                    self.logger.info("command: %s", " ".join(cmd))
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                    subprocess.run(cmd, env=env)
                    copy_file("expval", expval)

        return {
            "name": "expval_static:{}:compute".format(self.name),
            "actions": [action_compute],
            "targets": [str(self.expval)],
            "file_dep": [str(self.wave_function),
                         str(self.operator)],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_compute]
