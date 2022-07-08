from __future__ import annotations

import pickle
from os import PathLike
from typing import Any, Mapping, Sequence

import h5py
import numpy
import scipy.linalg
from numpy.typing import ArrayLike

from mlxtk.doit_compat import DoitAction
from mlxtk.dvr import DVRSpecification
from mlxtk.hashing import inaccurate_hash
from mlxtk.log import get_logger
from mlxtk.tasks.task import Task
from mlxtk.tools.reduced_density_matrix import compute_reduced_density_matrix

LOGGER = get_logger(__name__)


class ComputeReducedDensityMatrix(Task):
    def __init__(
        self,
        name: str,
        wave_function: PathLike,
        dvrs: Sequence[DVRSpecification],
        dofs_A: Sequence[int],
        basis_states: Mapping[int, Sequence[ArrayLike]] | None = None,
        diagonalize: bool = False,
        store_eigenvectors: bool = False,
        diag_cleanup: bool = False,
        threads: int = 1,
    ):
        self.name = name
        self.filename = name + ".h5"
        self.picklename = name + ".pickle"
        self.wave_function = wave_function
        self.dvrs = dvrs
        self.dofs_A = dofs_A
        self.basis_states = basis_states
        self.diagonalize = diagonalize
        self.store_eigenvectors = store_eigenvectors
        self.diag_cleanup = diag_cleanup
        self.threads = threads

    def task_write_parameters(self) -> dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: list[str]):
            del targets
            obj = [self.dvrs, self.dofs_A]
            if self.basis_states is not None:
                obj.append(
                    {
                        dof: [
                            inaccurate_hash(state) for state in self.basis_states[dof]
                        ]
                        for dof in self.basis_states
                    },
                )
            else:
                obj.append(None)
            obj += [self.diagonalize, self.store_eigenvectors, self.diag_cleanup]

            with open(self.picklename, "wb") as fptr:
                pickle.dump(obj, fptr, protocol=3)

        return {
            "name": f"reduced_density_matrix:{self.name}:write_parameters",
            "actions": [action_write_parameters],
            "targets": [self.picklename],
        }

    def task_compute(self) -> dict[str, Any]:
        @DoitAction
        def action_compute(targets: list[str]):
            del targets
            compute_reduced_density_matrix(
                self.wave_function,
                self.filename,
                self.dvrs,
                self.dofs_A,
                self.basis_states,
                self.threads,
            )

            if not self.diagonalize:
                return

            with h5py.File(self.filename, "r+") as fptr:
                time_steps, dim_A, _ = fptr["rho_A"].shape
                dset_evals = fptr.create_dataset(
                    "evals",
                    shape=(time_steps, dim_A),
                    dtype=numpy.float64,
                )
                dset_entropy = fptr.create_dataset(
                    "entropy",
                    shape=(time_steps,),
                    dtype=numpy.float64,
                )

                if self.store_eigenvectors:
                    dset_evecs = fptr.create_dataset(
                        "evecs",
                        shape=(time_steps, dim_A, dim_A),
                        dtype=numpy.complex128,
                    )

                for i in range(time_steps):
                    result = scipy.linalg.eigh(
                        fptr["rho_A"][i, :, :],
                        eigvals_only=not self.store_eigenvectors,
                    )
                    if self.store_eigenvectors:
                        dset_evals[i] = result[0]
                        dset_evecs[i] = result[1].T
                        evals = result[0]
                    else:
                        dset_evals[i] = result
                        evals = result

                    evals = evals[evals > 1e-19]
                    dset_entropy[i] = -numpy.sum(evals * numpy.log(evals))

                if self.diag_cleanup:
                    del fptr["rho_A"]
                    for dof in self.dofs_A:
                        del fptr[f"basis_states_{dof}"]

        return {
            "name": f"reduced_density_matrix:{self.name}:compute",
            "actions": [action_compute],
            "targets": [self.filename],
            "file_dep": [self.picklename],
        }

    def get_tasks_run(self):
        return [self.task_write_parameters, self.task_compute]
