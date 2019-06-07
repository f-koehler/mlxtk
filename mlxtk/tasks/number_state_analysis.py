import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import h5py

from .. import cwd
from ..doit_compat import DoitAction
from ..log import get_logger
from ..util import copy_file, make_path
from .task import Task


class NumberStateAnalysisStatic(Task):
    def __init__(self, wave_function: Union[str, Path],
                 basis: Union[str, Path], **kwargs):
        self.logger = get_logger(__name__ + ".NumberStateAnalysisStatic")

        self.wave_function = make_path(wave_function).with_suffix(".wfn")
        self.basis = make_path(basis).with_suffix(".wfn")
        self.result = kwargs.get(
            "name",
            self.wave_function.with_name(self.wave_function.stem + "_" +
                                         self.basis.stem)).with_suffix(".nsa")

        self.name = str(self.result.with_suffix(""))

    def task_compute(self) -> Dict[str, Any]:
        @DoitAction
        def action_compute(targets: List[str]):
            del targets

            wave_function = self.wave_function.resolve()
            basis = self.basis.resolve()
            result = self.result.resolve()

            with tempfile.TemporaryDirectory() as tmpdir:
                with cwd.WorkingDir(tmpdir):
                    self.logger.info("perform number state analysis (static)")
                    copy_file(wave_function, "restart")
                    copy_file(basis, "basis")
                    cmd = [
                        "qdtk_analysis.x", "-fixed_ns", "-rst_bra", "basis",
                        "-rst_ket", "restart", "-save", "result"
                    ]
                    self.logger.info("command: %s", " ".join(cmd))
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                    subprocess.run(cmd, env=env)
                    copy_file("result", result)

        return {
            "name":
            "number_state_analysis_static:{}:compute".format(self.name),
            "actions": [action_compute],
            "targets": [str(self.result)],
            "file_dep": [str(self.wave_function),
                         str(self.basis)]
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_compute]
