import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import numpy
from QDTK.SQR.WaveFunction import create_bosonic_sqr_wave_function_with_binary_tree

from mlxtk.doit_compat import DoitAction
from mlxtk.log import get_logger
from mlxtk.tasks.task import Task


class CreateSQRBosonicWaveFunction(Task):
    def __init__(
        self,
        name: str,
        orbital_list: Union[List[int], numpy.ndarray],
        N: int,
        L: int,
        initial_filling: Union[List[float], numpy.ndarray],
    ):
        self.logger = get_logger(__name__ + ".CreateSQRBosonicWaveFunction")
        self.name = name
        self.number_of_particles = N
        self.number_of_sites = L

        if isinstance(initial_filling, numpy.ndarray):
            self.initial_filling = initial_filling.tolist()
        else:
            self.initial_filling = initial_filling

        if isinstance(orbital_list, numpy.ndarray):
            self.orbital_list = orbital_list.astype(numpy.int64)
        else:
            self.orbital_list = orbital_list

        self.path = Path(name)
        self.path_pickle = Path(name + ".pickle")

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [
                self.name,
                self.orbital_list,
                self.number_of_particles,
                self.number_of_sites,
                self.initial_filling,
            ]
            with open(self.path_pickle, "wb") as fptr:
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": "wfn_sqr_bosonic:{}:write_parameters".format(self.name),
            "actions": [
                action_write_parameters,
            ],
            "targets": [self.path_pickle],
        }

    def task_write_wave_function(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_wave_function(targets: List[str]):
            create_bosonic_sqr_wave_function_with_binary_tree(
                self.orbital_list,
                self.number_of_particles,
                self.number_of_sites,
                numpy.array(self.initial_filling),
                self.path,
            )

        return {
            "name": "wfn_sqr_bosonic:{}:create".format(self.name),
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

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_write_wave_function]
