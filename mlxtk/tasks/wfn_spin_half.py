from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import numpy
from numpy.typing import NDArray
from QDTK.Spin.WaveFunction import create_spin_half_wave_function_with_binary_tree

from mlxtk.doit_compat import DoitAction
from mlxtk.log import get_logger
from mlxtk.tasks.task import Task


class CreateSpinHalfWaveFunction(Task):
    def __init__(
        self,
        name: str,
        orbital_list: list[int] | NDArray,
        L: int,
        spfs: list[NDArray],
    ):
        self.logger = get_logger(__name__ + ".CreateSpinHalfWaveFunction")
        self.name = name
        self.number_of_sites = L
        self.spfs = spfs

        self.orbital_list = orbital_list

        self.path = Path(name)
        self.path_pickle = Path(name + ".pickle")

    def task_write_parameters(self) -> dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: list[str]):
            del targets

            obj = [
                self.name,
                self.orbital_list,
                self.number_of_sites,
                self.spfs,
            ]
            with open(self.path_pickle, "wb") as fptr:
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": f"wfn_spin_half:{self.name}:write_parameters",
            "actions": [
                action_write_parameters,
            ],
            "targets": [self.path_pickle],
        }

    def task_write_wave_function(self) -> dict[str, Any]:
        @DoitAction
        def action_write_wave_function(targets: list[str]):
            create_spin_half_wave_function_with_binary_tree(
                self.orbital_list,
                self.number_of_sites,
                self.spfs,
                self.path,
            )

        return {
            "name": f"wfn_spin_half:{self.name}:create",
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
