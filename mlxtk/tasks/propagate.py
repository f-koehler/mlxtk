import copy
import os
import pickle
import shutil
import subprocess
from typing import Any, Callable, Dict, List, Tuple

from .. import cwd
from ..log import get_logger
from ..doit_compat import DoitAction
from .task import Task

LOGGER = get_logger(__name__)

FLAG_TYPES = {
    "MBop_apply": bool,
    "atol": float,
    "cont": bool,
    "dt": float,
    "eig_index": int,
    "eig_tot": int,
    "exact_diag": bool,
    "exproj": bool,
    "gauge": str,
    "gramschmidt": bool,
    "improved_relax": bool,
    "itg": str,
    "nstep_diag": int,
    "opr": str,
    "pruning_eom": str,
    "pruning_method": str,
    "pruning_threshold": float,
    "psi": bool,
    "reg": float,
    "relax": bool,
    "resetnorm": bool,
    "rst": str,
    "rstzero": bool,
    "rtol": float,
    "stat_energ_tol": float,
    "stat_npop_tol": float,
    "statsteps": int,
    "tfinal": float,
    "timing": bool,
    "transMat": bool,
    "zvode_mf": int,
}

DEFAULT_FLAGS = {
    "MBop_apply": False,
    "atol": 1e-12,
    "cont": False,
    "dt": 0.1,
    "exact_diag": False,
    "exproj": True,
    "gramschmidt": False,
    "improved_relax": False,
    "itg": "zvode",
    "psi": False,
    "reg": 1e-8,
    "relax": False,
    "resetnorm": False,
    "rstzero": True,
    "rtol": 1e-12,
    "tfinal": 1.0,
    "timing": True,
    "transMat": False,
}


def create_flags(**kwargs) -> Tuple[Dict[str, Any], List[str]]:
    flags = copy.copy(DEFAULT_FLAGS)

    for flag in FLAG_TYPES:
        if flag not in kwargs:
            continue
        flags[flag] = kwargs[flag]

    if flags["relax"]:
        flags["statsteps"] = flags.get("statsteps", 50)

    if flags["improved_relax"]:
        flags["statsteps"] = flags.get("statsteps", 40)
        flags["eig_index"] = flags.get("eig_index", 1)
        flags["nstep_diag"] = flags.get("nstep_diag", 50)

    if flags["itg"] == "zvode":
        flags["zvode_mf"] = flags.get("zvode_mf", 10)

    flag_list = []
    for flag in flags:
        if FLAG_TYPES[flag] == bool:
            if flags[flag]:
                flag_list.append("-" + flag)
        elif FLAG_TYPES[flag] == str:
            flag_list += ["-" + flag, flags[flag]]
        else:
            flag_list += ["-" + flag, str(flags[flag])]

    return flags, flag_list


class Propagate(Task):
    def __init__(self, name: str, wave_function: str, hamiltonian: str,
                 **kwargs):
        self.name = name
        self.wave_function = wave_function
        self.hamiltonian = hamiltonian

        self.flags, self.flag_list = create_flags(**kwargs)
        self.flag_list += ["-rst", "initial.wfn", "-opr", "hamiltonian.mb_opr"]

        if self.flags["relax"]:
            self.basename = "relaxation"
        elif self.flags["improved_relax"]:
            self.basename = "improved_relaxation"
        elif self.flags["exact_diag"]:
            self.basename = "exact_diagonalization"
        else:
            self.basename = "propagation"

        # compute required paths
        self.path_pickle = self.name + ".prop_pickle"
        self.path_output = os.path.join(self.name, "output")
        self.path_gpop = os.path.join(self.name, "gpop")
        self.path_natpop = os.path.join(self.name, "natpop")
        self.path_restart = os.path.join(self.name, "restart")
        self.path_final = os.path.join(self.name, "final.wfn")
        self.path_psi = os.path.join(self.name, "psi")
        self.path_energies = os.path.join(self.name, "eigenenergies")
        self.path_eigenvectors = os.path.join(self.name, "eigenvectors")
        self.path_wfn = self.wave_function + ".wfn"
        self.path_opr = self.hamiltonian + ".mb_opr"
        self.path_wfn_tmp = os.path.join(self.name, "initial.wfn")
        self.path_opr_tmp = os.path.join(self.name, "hamiltonian.mb_opr")

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [self.name, self.wave_function, self.hamiltonian, self.flags]
            with open(self.path_pickle, "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "{}:{}:write_parameters".format(self.basename, self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
        }

    def task_propagate(self) -> Dict[str, Any]:
        @DoitAction
        def action_create_dir(targets: List[str]):
            del targets

            if not os.path.exists(self.name):
                LOGGER.info("create propagation directory")
                os.makedirs(self.name)

        @DoitAction
        def action_copy_initial_wave_function(targets: List[str]):
            del targets
            LOGGER.info("copy initial wave function")
            shutil.copy2(self.path_wfn, self.path_wfn_tmp)

        @DoitAction
        def action_copy_hamiltonian(targets: List[str]):
            del targets
            LOGGER.info("copy Hamiltonian")
            shutil.copy2(self.path_opr, self.path_opr_tmp)

        @DoitAction
        def action_run(targets: List[str]):
            del targets
            with cwd.WorkingDir(os.path.abspath(self.name)):
                LOGGER.info("propagate wave function")
                cmd = ["qdtk_propagate.x"] + self.flag_list
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                LOGGER.info("command: %s", " ".join(cmd))
                subprocess.run(cmd, env=env)

        @DoitAction
        def action_remove_hamiltonian(targets: List[str]):
            del targets
            LOGGER.info("remove Hamiltonian")
            os.remove(self.path_opr_tmp)

        @DoitAction
        def action_remove_initial_wave_function(targets: List[str]):
            del targets
            LOGGER.info("remove initial wave function")
            os.remove(self.path_wfn_tmp)

        @DoitAction
        def action_move_final_wave_function(targets: List[str]):
            del targets
            LOGGER.info("move final wave function %s -> %s", self.path_restart,
                        self.path_final)
            shutil.move(self.path_restart, self.path_final)

        if self.flags["exact_diag"]:
            targets = [self.path_energies, self.path_eigenvectors]
            actions = [
                action_create_dir,
                action_copy_initial_wave_function,
                action_copy_hamiltonian,
                action_run,
                action_remove_hamiltonian,
                action_remove_initial_wave_function,
            ]
        else:
            targets = [
                self.path_output,
                self.path_gpop,
                self.path_natpop,
                self.path_final,
            ]
            if self.flags["psi"]:
                targets.append(self.path_psi)

            actions = [
                action_create_dir,
                action_copy_initial_wave_function,
                action_copy_hamiltonian,
                action_run,
                action_remove_hamiltonian,
                action_remove_initial_wave_function,
                action_move_final_wave_function,
            ]

        return {
            "name": "{}:{}:run".format(self.basename, self.name),
            "actions": actions,
            "targets": targets,
            "file_dep": [self.path_pickle, self.path_wfn, self.path_opr],
            "verbosity": 2,
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_propagate]

    def get_tasks_clean(self) -> List[Callable[[], Dict[str, Any]]]:
        return []


class Relax(Propagate):
    def __init__(self, name: str, wave_function: str, hamiltonian: str,
                 **kwargs):
        kwargs["relax"] = True
        kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
        kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
        kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)

        super().__init__(name, wave_function, hamiltonian, **kwargs)


class ImprovedRelax(Propagate):
    def __init__(self, name: str, wave_function: str, hamiltonian: str,
                 eig_index: int, **kwargs):
        kwargs["improved_relax"] = True
        kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
        kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
        kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)
        kwargs["nstep_diag"] = kwargs.get("nstep_diag", 50)
        kwargs["stat_steps"] = kwargs.get("stat_steps", 80)
        kwargs["eig_index"] = eig_index

        super().__init__(name, wave_function, hamiltonian, **kwargs)


class Diagonalize(Propagate):
    def __init__(self, name: str, wave_function: str, hamiltonian: str,
                 number_of_states: int, **kwargs):
        kwargs["exact_diag"] = True
        kwargs["eig_tot"] = number_of_states

        super().__init__(name, wave_function, hamiltonian, **kwargs)
