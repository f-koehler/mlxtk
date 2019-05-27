from pathlib import Path
from typing import Any, Callable, Dict, List, Union

import h5py
import numpy

from ..doit_compat import DoitAction
from ..inout.spectrum import read_spectrum, write_spectrum
from ..tools.diagonalize import diagonalize_1b_operator
from .operator import OperatorSpecification
from .task import Task


class ComputeSpectrum(Task):
    def __init__(self, hamiltonian_1b: Union[str, OperatorSpecification],
                 num_spfs: int):
        self.name = hamiltonian_1b
        self.num_spfs = num_spfs

        self.path = Path(self.name + ".spectrum.hdf5")
        self.path_matrix = Path(self.name + ".opr_mat.hdf5")
        self.path_matrix_hash = Path(self.name + ".opr_mat.hash")

    def task_check_num_spfs(self) -> Dict[str, Any]:
        @DoitAction
        def action_check_num_spfs(targets):
            del targets

            if self.path.exists():
                energies, _ = read_spectrum(self.path)

                if len(energies) != self.num_spfs:
                    self.path.unlink()

        return {
            "name": "spectrum:{}:check_num_spfs".format(self.name),
            "actions": [action_check_num_spfs]
        }

    def task_compute(self):
        @DoitAction
        def action_compute(targets):
            del targets

            with h5py.File(self.path_matrix, "r") as fp:
                matrix = fp["matrix"][:, :]

            energies, spfs = diagonalize_1b_operator(matrix, self.num_spfs)
            spfs_arr = numpy.array(spfs)
            write_spectrum(self.path, energies, spfs_arr)

        return {
            "name": "spectrum:{}:compute".format(self.name),
            "actions": [action_compute],
            "targets": [self.path],
            "file_dep": [self.path_matrix],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_check_num_spfs, self.task_compute]
