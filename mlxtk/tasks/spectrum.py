import os

import h5py
import numpy

from ..doit_compat import DoitAction
from ..inout.spectrum import read_spectrum, write_spectrum
from ..tools.diagonalize import diagonalize_1b_operator


def compute_spectrum(hamiltonian_1b: str, num_spfs: int):
    path = hamiltonian_1b + ".spectrum.hdf5"
    path_matrix = hamiltonian_1b + ".opr_mat.hdf5"

    def task_check_num_spfs():
        @DoitAction
        def action_check_num_spfs(targets):
            del targets

            if os.path.exists(path):
                energies, _ = read_spectrum(path)

                if len(energies) != num_spfs:
                    os.remove(path)

        return {
            "name": "spectrum:{}:check_num_spfs".format(hamiltonian_1b),
            "actions": [action_check_num_spfs]
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
            "name": "spectrum:{}:compute".format(hamiltonian_1b),
            "actions": [action_compute],
            "targets": [path],
            "file_dep": [path_matrix]
        }

    return [task_check_num_spfs, task_compute]
