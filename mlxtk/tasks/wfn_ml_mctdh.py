from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Union, Optional

import numpy
from numpy.typing import NDArray
from QDTK.Wavefunction import Wavefunction

from mlxtk.doit_compat import DoitAction
from mlxtk.log import get_logger
from mlxtk.tasks.task import Task


class CreateMLMCTDHWaveFunction(Task):
    def __init__(
        self,
        name: str,
        tape: list[int],
        spfs: list[NDArray],
        shake: Optional[dict[str,Any]] = None
    ):
        self.logger = get_logger(__name__ + ".CreateMLMCTDHWaveFunction")
        self.name = name
        self.tape = tape
        self.spfs = spfs
        self.shake = None

        if shake is not None:
            self.shake = {
                "strength": shake.get("strength", 0.1),
                "primitive": shake.get("primitive", False),
                "onlytop": shake.get("onlytop", False),
                "cmplx": shake.get("cmplx", False),
                "seed": shake.get("seed", 0),
            }

        self.path = Path(name)
        self.path_pickle = Path(name + ".pickle")

    def task_write_parameters(self) -> dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: list[str]):
            del targets

            obj = [
                self.name,
                self.tape,
                self.spfs,
            ]
            if self.shake is not None:
                obj.append(["shake", self.shake["primitive"], self.shake["onlytop"], self.shake["cmplx"], self.shake["seed"],])
            with open(self.path_pickle, "wb") as fptr:
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": f"wfn_mlmctdh:{self.name}:write_parameters",
            "actions": [
                action_write_parameters,
            ],
            "targets": [self.path_pickle],
        }

    def task_write_wave_function(self) -> dict[str, Any]:
        @DoitAction
        def action_write_wave_function(targets: list[str]):
            wfn = Wavefunction(tape=self.tape)
            wfn.build_primitive_SPF(self.spfs)
            wfn.build_coefs()
            if self.shake is not None:
                wfn.shake(**self.shake)
            wfn.createWfnFile(self.path)

        return {
            "name": f"wfn_mlmctdh:{self.name}:create",
            "actions": [
                action_write_wave_function,
            ],
            "targets": [
                self.path,
            ],
            "file_dep": [
                self.path_pickle,
            ],
            "verbosity": 2,
        }

    def get_tasks_run(self) -> list[Callable[[], dict[str, Any]]]:
        return [self.task_write_parameters, self.task_write_wave_function]
