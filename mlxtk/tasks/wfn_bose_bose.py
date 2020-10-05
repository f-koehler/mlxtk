import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import h5py
import numpy
from QDTK.Wavefunction import Wavefunction as WaveFunction

from mlxtk.doit_compat import DoitAction
from mlxtk.tasks.task import Task
from mlxtk.tools.diagonalize import diagonalize_1b_operator
from mlxtk.tools.wave_function import save_wave_function


class CreateBoseBoseWaveFunction(Task):
    def __init__(
        self,
        name: str,
        hamiltonian_1b_A: str,
        hamiltonian_1b_B: str,
        num_particles_A: int,
        num_particles_B: int,
        num_sbs_A: int,
        num_sbs_B: int,
        num_spfs_A: int,
        num_spfs_B: int,
        ns_A: Optional[Union[numpy.ndarray, List]] = None,
        ns_B: Optional[Union[numpy.ndarray, List]] = None,
    ):
        self.name = name
        self.hamiltonian_1b_A = hamiltonian_1b_A
        self.hamiltonian_1b_B = hamiltonian_1b_B
        self.num_particles_A = num_particles_A
        self.num_particles_B = num_particles_B
        self.num_sbs_A = num_sbs_A
        self.num_sbs_B = num_sbs_B
        self.num_spfs_A = num_spfs_A
        self.num_spfs_B = num_spfs_B

        if ns_A is None:
            self.ns_A = numpy.zeros(self.num_spfs_A, dtype=numpy.int64)
            self.ns_A[0] = self.num_particles_A
        else:
            self.ns_A = numpy.copy(numpy.array(ns_A))

        if ns_B is None:
            self.ns_B = numpy.zeros(self.num_spfs_B, dtype=numpy.int64)
            self.ns_B[0] = self.num_particles_B
        else:
            self.ns_B = numpy.copy(numpy.array(ns_B))

        self.path = Path(name)
        self.path_pickle = Path(name + ".pickle")
        self.path_basis_A = Path(name + "_A_basis.h5")
        self.path_basis_B = Path(name + "_B_basis.h5")
        self.path_matrix_A = Path(self.hamiltonian_1b_A + "_mat.h5")
        self.path_matrix_B = Path(self.hamiltonian_1b_B + "_mat.h5")

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [
                self.name,
                self.hamiltonian_1b_A,
                self.hamiltonian_1b_B,
                self.num_particles_A,
                self.num_particles_B,
                self.num_sbs_A,
                self.num_sbs_B,
                self.num_spfs_A,
                self.num_spfs_B,
                self.ns_A.tolist(),
                self.ns_B.tolist(),
            ]

            with open(self.path_pickle, "wb") as fptr:
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": "wfn_bose_bose:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
        }

    def task_write_wave_function(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_wave_function(targets: List[str]) -> Dict[str, Any]:
            with h5py.File(self.path_matrix_A, "r") as fptr:
                matrix_A = fptr["matrix"][:, :]

            with h5py.File(self.path_matrix_B, "r") as fptr:
                matrix_B = fptr["matrix"][:, :]

            energies_A, spfs_A = diagonalize_1b_operator(matrix_A, self.num_spfs_A)
            spfs_arr_A = numpy.array(spfs_A)

            energies_B, spfs_B = diagonalize_1b_operator(matrix_B, self.num_spfs_B)
            spfs_arr_B = numpy.array(spfs_B)

            with h5py.File(self.path_basis_A, "w") as fptr:
                fptr.create_dataset(
                    "energies", (self.num_spfs_A,), dtype=numpy.float64
                )[:] = energies_A
                fptr.create_dataset("spfs", spfs_arr_A.shape, dtype=numpy.complex128)[
                    :, :
                ] = spfs_arr_A

            with h5py.File(self.path_basis_B, "w") as fptr:
                fptr.create_dataset(
                    "energies", (self.num_spfs_B,), dtype=numpy.float64
                )[:] = energies_B
                fptr.create_dataset("spfs", spfs_arr_B.shape, dtype=numpy.complex128)[
                    :, :
                ] = spfs_arr_B

            n_A = matrix_A.shape[0]
            n_B = matrix_B.shape[0]
            assert n_A == n_B

            tape = (
                -10,
                2,
                0,
                self.num_sbs_A,
                self.num_sbs_B,
                -1,
                1,
                self.num_particles_A,
                1,
                self.num_spfs_A,
                -1,
                1,
                1,
                0,
                n_A,
                0,
                0,
                -1,
                2,
                self.num_particles_B,
                1,
                self.num_spfs_B,
                -1,
                1,
                1,
                0,
                n_B,
                -2,
            )

            wfn = WaveFunction(tape=tape)
            wfn.init_coef_multi_spec(
                2, [self.ns_A, self.ns_B], [spfs_A, spfs_B], 1e-15, 1e-15, full_spf=True
            )
            save_wave_function(self.path, wfn)

        return {
            "name": "wfn_bose_bose:{}:create".format(self.name),
            "actions": [
                action_write_wave_function,
            ],
            "targets": [
                self.path,
                self.path_basis_A,
                self.path_basis_B,
            ],
            "file_dep": [self.path_pickle, self.path_matrix_A, self.path_matrix_B],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_write_wave_function]
