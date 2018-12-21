import pickle
from typing import Any, Dict, Iterable

import h5py
import numpy

from QDTK.Wavefunction import Wavefunction as WaveFunction

from ..doit_compat import DoitAction
from ..dvr import DVRSpecification
from ..tools.diagonalize import diagonalize_1b_operator
from ..tools.wave_function import (add_momentum, add_momentum_split,
                                   load_wave_function, save_wave_function)


def create_mctdhb_wave_function(name,
                                hamiltonian_1b,
                                num_particles,
                                num_spfs,
                                number_state=None):
    path_pickle = name + ".wfn_pickle"

    if number_state is None:
        number_state = numpy.zeros(num_spfs, dtype=numpy.int64)
        number_state[0] = num_particles
    elif not isinstance(number_state, numpy.ndarray):
        number_state = numpy.array(number_state)

    def task_write_parameters():
        @DoitAction
        def action_write_parameters(targets):
            obj = [
                name, hamiltonian_1b, num_particles, num_spfs,
                number_state.tolist()
            ]
            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "wfn_mctdhb:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_write_wave_function():
        path = name + ".wfn"
        path_basis = name + ".wfn_basis.hdf5"
        path_matrix = hamiltonian_1b + ".opr_mat.hdf5"

        @DoitAction
        def action_write_wave_function(targets):
            with h5py.File(path_matrix, "r") as fp:
                matrix = fp["matrix"][:, :]

            energies, spfs = diagonalize_1b_operator(matrix, num_spfs)
            spfs_arr = numpy.array(spfs)

            with h5py.File(targets[1], "w") as fp:
                dset = fp.create_dataset(
                    "energies", (num_spfs, ),
                    dtype=numpy.float64,
                    compression="gzip")
                dset[:] = energies

                dset = fp.create_dataset(
                    "spfs",
                    spfs_arr.shape,
                    dtype=numpy.complex128,
                    compression="gzip")
                dset[:, :] = spfs_arr

            grid_points = matrix.shape[0]
            tape = (-10, num_particles, +1, num_spfs, -1, 1, 1, 0, grid_points,
                    -2)
            wfn = WaveFunction(tape=tape)
            wfn.init_coef_sing_spec_B(
                number_state, spfs, 1e-15, 1e-15, full_spf=True)

            save_wave_function(targets[0], wfn)

        return {
            "name": "wfn_mctdhb:{}:create".format(name),
            "actions": [action_write_wave_function],
            "targets": [path, path_basis],
            "file_dep": [path_pickle, path_matrix],
        }

    return [task_write_parameters, task_write_wave_function]


def mctdhb_add_momentum(name: str,
                        initial: str,
                        momentum: float,
                        grid: DVRSpecification):
    path_pickle = name + ".wfn_pickle"

    def task_write_parameters() -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: Iterable[str]):
            del targets
            obj = [name, initial, momentum]
            with open(path_pickle, "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "wfn_mctdhb_add_momentum:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_add_momentum() -> Dict[str, Any]:
        path = name + ".wfn"
        initial_path = initial + ".wfn"

        @DoitAction
        def action_add_momentum(targets: Iterable[str]):
            del targets

            # pylint: disable=protected-access

            wfn = load_wave_function(initial_path)
            wfn.tree._topNode._pgrid[0] = grid.get_x()
            add_momentum(wfn, momentum)
            save_wave_function(path, wfn)

        return {
            "name": "wfn_mctdhb_add_momentum:{}:add_momentum".format(name),
            "actions": [action_add_momentum],
            "targets": [path],
            "file_dep": [path_pickle],
        }

    return [task_write_parameters, task_add_momentum]


def mctdhb_add_momentum_split(name: str,
                              initial: str,
                              momentum: float,
                              x0: float,
                              grid: DVRSpecification):
    path_pickle = name + ".wfn_pickle"

    def task_write_parameters() -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: Iterable[str]):
            del targets
            obj = [name, initial, momentum, x0]
            with open(path_pickle, "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name":
            "wfn_mctdhb_add_momentum_split:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_add_momentum_split() -> Dict[str, Any]:
        path = name + ".wfn"
        initial_path = initial + ".wfn"

        @DoitAction
        def action_add_momentum_split(targets: Iterable[str]):
            del targets

            # pylint: disable=protected-access

            wfn = load_wave_function(initial_path)
            wfn.tree._topNode._pgrid[0] = grid.get_x()
            add_momentum_split(wfn, momentum, x0)
            save_wave_function(path, wfn)

        return {
            "name":
            "wfn_mctdhb_add_momentum_split:{}:add_momentum".format(name),
            "actions": [action_add_momentum_split],
            "targets": [path],
            "file_dep": [path_pickle],
        }

    return [task_write_parameters, task_add_momentum_split]
