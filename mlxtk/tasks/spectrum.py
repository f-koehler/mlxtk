import os
from typing import Union

import h5py
import numpy

from ..doit_compat import DoitAction
from ..hashing import inaccurate_hash
from ..inout.spectrum import read_spectrum, write_spectrum
from ..tools.diagonalize import diagonalize_1b_operator
from ..tools.operator import get_operator_matrix
from .operator import OperatorSpecification


def compute_spectrum(hamiltonian_1b: Union[str, OperatorSpecification],
                     num_spfs: int,
                     hamiltonian_1b_name: str = None):

    if isinstance(hamiltonian_1b, OperatorSpecification):
        if hamiltonian_1b_name is None:
            raise ValueError(
                "hamiltonian_1b_name is required when using an OperatorSpecification"
            )
        else:
            name = hamiltonian_1b_name
    else:
        name = hamiltonian_1b

    path = name + ".spectrum.hdf5"
    path_matrix = name + ".opr_mat.hdf5"
    path_matrix_hash = name + ".opr_mat.hash"

    def task_check_num_spfs():
        @DoitAction
        def action_check_num_spfs(targets):
            del targets

            if os.path.exists(path):
                energies, _ = read_spectrum(path)

                if len(energies) != num_spfs:
                    os.remove(path)

        return {
            "name": "spectrum:{}:check_num_spfs".format(name),
            "actions": [action_check_num_spfs]
        }

    matrix = None

    def task_matrix_hash():
        @DoitAction
        def action_matrix_hash(targets):
            del targets
            nonlocal matrix

            matrix = get_operator_matrix(hamiltonian_1b.get_operator())
            with open(path_matrix_hash, "w") as fp:
                hsh = inaccurate_hash(matrix)
                fp.write(hsh)

        return {
            "name": "spectrum:{}:write_matrix_hash".format(name),
            "actions": [action_matrix_hash],
        }

    def task_compute():
        @DoitAction
        def action_compute(targets):
            del targets

            with h5py.File(path_matrix, "r") as fp:
                matrix = fp["matrix"][:, :]

            energies, spfs = diagonalize_1b_operator(matrix, num_spfs)
            spfs_arr = numpy.array(spfs)
            write_spectrum(path, energies, spfs_arr)

        return {
            "name": "spectrum:{}:compute".format(name),
            "actions": [action_compute],
            "targets": [path],
            "file_dep": [path_matrix],
        }

    def task_compute_spec():
        @DoitAction
        def action_compute(targets):
            del targets
            nonlocal matrix

            energies, spfs = diagonalize_1b_operator(matrix, num_spfs)
            spfs_arr = numpy.array(spfs)
            write_spectrum(path, energies, spfs_arr)

        return {
            "name": "spectrum:{}:compute_from_spec".format(name),
            "actions": [action_compute],
            "targets": [path],
            "file_dep": [path_matrix_hash],
        }

    if isinstance(hamiltonian_1b, OperatorSpecification):
        return [task_check_num_spfs, task_matrix_hash, task_compute_spec]
    else:
        return [task_check_num_spfs, task_compute]
