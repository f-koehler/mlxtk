import gzip
import io
import pickle

import h5py
import numpy

from QDTK.Wavefunction import Wavefunction as WaveFunction

from ..tools.diagonalize import diagonalize_1b_operator


def create_mctdhb_wave_function(name, hamiltonian_1b, num_particles, num_spfs):
    path_pickle = name + ".wfn_pickle"

    def task_write_parameters():
        def action_write_parameters(targets):
            obj = [name, hamiltonian_1b, num_particles, num_spfs]
            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "create_mctdhb_wave_function:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_write_wave_function():
        path = name + ".wfn.gz"
        path_basis = name + ".wfn_basis.hdf5"
        path_matrix = hamiltonian_1b + ".opr_mat.hdf5"

        def action_write_wave_function(targets):
            with h5py.File(path_matrix, "r") as fp:
                matrix = fp["matrix"][:, :]

            energies, spfs = diagonalize_1b_operator(matrix, num_spfs)
            spfs_arr = numpy.array(spfs)

            with h5py.File(targets[1], "w") as fp:
                dset = fp.create_dataset(
                    "energies", (num_spfs,), dtype=numpy.float64, compression="gzip"
                )
                dset[:] = energies

                dset = fp.create_dataset(
                    "spfs", spfs_arr.shape, dtype=numpy.complex128, compression="gzip"
                )
                dset[:, :] = spfs_arr

            grid_points = matrix.shape[0]
            tape = (-10, num_particles, +1, num_spfs, -1, 1, 1, 0, grid_points, -2)
            ns = numpy.zeros(num_spfs)
            ns[0] = num_particles
            wfn = WaveFunction(tape=tape)
            wfn.init_coef_sing_spec_B(ns, spfs, 1e-15, 1e-15, full_spf=True)

            with gzip.open(targets[0], "wb") as fp:
                with io.StringIO() as sio:
                    wfn.createWfnFile(sio)
                    fp.write(sio.getvalue().encode())

        return {
            "name": "create_mctdhb_wave_function:{}:write_wave_function".format(name),
            "actions": [action_write_wave_function],
            "targets": [path, path_basis],
            "file_dep": [path_pickle, path_matrix],
        }

    return [task_write_parameters, task_write_wave_function]
