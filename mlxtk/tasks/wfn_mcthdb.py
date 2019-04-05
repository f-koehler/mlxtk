import pickle
from typing import Any, Callable, Dict, List

import h5py
import numpy

from mlxtk.doit_compat import DoitAction
from mlxtk.dvr import DVRSpecification
from mlxtk.tasks.task import Task
from mlxtk.tools.diagonalize import diagonalize_1b_operator
from mlxtk.tools.wave_function import (add_momentum, add_momentum_split,
                                       load_wave_function, save_wave_function,
                                       get_spfs)
from mlxtk.log import get_logger
from QDTK.Wavefunction import Wavefunction as WaveFunction

LOGGER = get_logger(__name__)


class MCTDHBCreateWaveFunction(Task):
    def __init__(self,
                 name: str,
                 hamiltonian_1b: str,
                 num_particles: int,
                 num_spfs: int,
                 number_state: numpy.ndarray = None):
        self.name = name
        self.number_of_particles = num_particles
        self.number_of_spfs = num_spfs
        self.hamiltonian_1b = hamiltonian_1b

        if number_state is None:
            self.number_state = numpy.zeros(
                self.number_of_spfs, dtype=numpy.int64)
            self.number_state[0] = self.number_of_particles
        else:
            self.number_state = numpy.copy(number_state)

        self.path = name + ".wfn"
        self.path_pickle = name + ".wfn_pickle"
        self.path_basis = name + ".wfn_basis.hdf5"
        self.path_matrix = self.hamiltonian_1b + ".opr_mat.hdf5"

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [
                self.name, self.hamiltonian_1b, self.number_of_particles,
                self.number_of_spfs,
                self.number_state.tolist()
            ]
            with open(self.path_pickle, "wb") as fptr:
                pickle.dump(obj, fptr)

        return {
            "name": "wfn_mctdhb:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
        }

    def task_write_wave_function(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_wave_function(targets: List[str]):
            with h5py.File(self.path_matrix, "r") as fptr:
                matrix = fptr["matrix"][:, :]

            energies, spfs = diagonalize_1b_operator(matrix,
                                                     self.number_of_spfs)
            spfs_arr = numpy.array(spfs)

            with h5py.File(targets[1], "w") as fptr:
                dset = fptr.create_dataset(
                    "energies", (self.number_of_spfs, ),
                    dtype=numpy.float64,
                    compression="gzip")
                dset[:] = energies

                dset = fptr.create_dataset(
                    "spfs",
                    spfs_arr.shape,
                    dtype=numpy.complex128,
                    compression="gzip")
                dset[:, :] = spfs_arr

            grid_points = matrix.shape[0]
            tape = (-10, self.number_of_particles, +1, self.number_of_spfs, -1,
                    1, 1, 0, grid_points, -2)
            wfn = WaveFunction(tape=tape)
            wfn.init_coef_sing_spec_B(
                self.number_state, spfs, 1e-15, 1e-15, full_spf=True)

            save_wave_function(self.path, wfn)

        return {
            "name": "wfn_mctdhb:{}:create".format(self.name),
            "actions": [action_write_wave_function],
            "targets": [self.path, self.path_basis],
            "file_dep": [self.path_pickle, self.path_matrix],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_write_wave_function]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return []


class MCTDHBAddMomentum(Task):
    def __init__(self, name: str, initial: str, momentum: float,
                 grid: DVRSpecification):
        self.name = name
        self.initial = initial
        self.momentum = momentum
        self.grid = grid

        self.path = self.name + ".wfn"
        self.path_initial = self.initial + ".wfn"
        self.path_pickle = self.name + ".wfn_pickle"

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [self.name, self.initial, self.momentum]
            with open(self.path_pickle, "wb") as fptr:
                pickle.dump(obj, fptr)

        return {
            "name":
            "wfn_mctdhb_add_momentum:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
        }

    def task_add_momentum(self) -> Dict[str, Any]:
        @DoitAction
        def action_add_momentum(targets: List[str]):
            del targets

            # pylint: disable=protected-access

            wfn = load_wave_function(self.path_initial)
            wfn.tree._topNode._pgrid[0] = self.grid.get_x()
            add_momentum(wfn, self.momentum)
            save_wave_function(self.path, wfn)

        return {
            "name":
            "wfn_mctdhb_add_momentum:{}:add_momentum".format(self.name),
            "actions": [action_add_momentum],
            "targets": [self.path],
            "file_dep": [self.path_pickle],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_add_momentum]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return []


class MCTDHBAddMomentumSplit(Task):
    def __init__(self, name: str, initial: str, momentum: float, x0: float,
                 grid: DVRSpecification):
        self.name = name
        self.initial = initial
        self.momentum = momentum
        self.x0 = x0
        self.grid = grid

        self.path = name + ".wfn"
        self.path_initial = self.initial + ".wfn"
        self.path_pickle = self.name + ".wfn_pickle"

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [self.name, self.initial, self.momentum, self.x0]
            with open(self.path_pickle, "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name":
            "wfn_mctdhb_add_momentum_split:{}:write_parameters".format(
                self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
        }

    def task_add_momentum_split(self) -> Dict[str, Any]:
        @DoitAction
        def action_add_momentum_split(targets: List[str]):
            del targets

            # pylint: disable=protected-access

            wfn = load_wave_function(self.path_initial)
            wfn.tree._topNode._pgrid[0] = self.grid.get_x()
            add_momentum_split(wfn, self.momentum, self.x0)
            save_wave_function(self.path, wfn)

        return {
            "name":
            "wfn_mctdhb_add_momentum_split:{}:add_momentum".format(self.name),
            "actions": [action_add_momentum_split],
            "targets": [self.path],
            "file_dep": [self.path_pickle],
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_add_momentum_split]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return []


class MCTDHBExtendGrid(Task):
    def __init__(self,
                 name: str,
                 initial: str,
                 nleft: int,
                 nright: int,
                 value: complex = 0.):
        self.name = name
        self.initial = initial
        self.nleft = nleft
        self.nright = nright
        self.value = value

        self.path = name + ".wfn"
        self.path_initial = self.initial + ".wfn"
        self.path_pickle = self.name + ".wfn_pickle"

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [
                self.name, self.initial, self.nleft, self.nright, self.value
            ]
            with open(self.path_pickle, "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name":
            "wfn_mctdhb_extend_grid:{}:write_parameters".format(self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle]
        }

    def task_extend_grid(self) -> Dict[str, Any]:
        @DoitAction
        def action_extend_grid(targets: List[str]):
            del targets

            # pylint: disable=protected-access
            wfn = load_wave_function(self.path_initial)
            tape = list(wfn._tape)
            tape[8] += self.nleft + self.nright
            wfn_new = WaveFunction(tape=tuple(tape))
            spfs = get_spfs(wfn)
            for i, spf in enumerate(spfs):
                spfs[i] = numpy.pad(
                    spf, (self.nleft, self.nright),
                    "constant",
                    constant_values=(self.value, self.value))

            N = wfn._tape[1]
            m = wfn._tape[3]

            number_state = numpy.zeros(m, dtype=numpy.int64)
            number_state[0] = N
            wfn_new.init_coef_sing_spec_B(
                number_state, spfs, 1e-15, 1e-15, full_spf=True)

            wfn_new.PSI[0:wfn_new.tree._subnodes[0].
                        _z0] = wfn.PSI[0:wfn.tree._subnodes[0]._z0]

            save_wave_function(self.path, wfn_new)

        return {
            "name": "wfn_mctdhb_extend_grid:{}:extend_grid".format(self.name),
            "actions": [action_extend_grid],
            "targets": [self.path],
            "file_dep": [self.path_pickle]
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_extend_grid]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return []
