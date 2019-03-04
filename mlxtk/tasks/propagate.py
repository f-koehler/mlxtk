import copy
import os
import pickle
import shutil
import subprocess
from typing import Any, Callable, Dict, List

from .. import cwd, log
from ..doit_compat import DoitAction

LOGGER = log.get_logger(__name__)

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


def create_flags(**kwargs):
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


def propagate(name: str, wave_function: str, hamiltonian: str,
              **kwargs) -> List[Callable[[], Dict[str, Any]]]:
    flags, flag_list = create_flags(**kwargs)
    flag_list += ["-rst", "initial.wfn", "-opr", "hamiltonian.mb_opr"]

    path_pickle = name + ".prop_pickle"

    if flags["relax"]:
        basename = "relaxation"
    elif flags["improved_relax"]:
        basename = "improved_relaxation"
    elif flags["exact_diag"]:
        basename = "exact_diagonalization"
    else:
        basename = "propagation"

    def task_write_parameters() -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            obj = [name, wave_function, hamiltonian, flags]

            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "{}:{}:write_parameters".format(basename, name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_propagate() -> Dict[str, Any]:
        path_output = os.path.join(name, "output")
        path_gpop = os.path.join(name, "gpop")
        path_natpop = os.path.join(name, "natpop")
        path_restart = os.path.join(name, "restart")
        path_final = os.path.join(name, "final.wfn")
        path_psi = os.path.join(name, "psi")
        path_energies = os.path.join(name, "eigenenergies")
        path_eigenvectors = os.path.join(name, "eigenvectors")

        path_wfn = wave_function + ".wfn"
        path_opr = hamiltonian + ".mb_opr"
        path_wfn_tmp = os.path.join(name, "initial.wfn")
        path_opr_tmp = os.path.join(name, "hamiltonian.mb_opr")

        @DoitAction
        def action_create_dir(targets: List[str]):
            del targets
            if not os.path.exists(name):
                LOGGER.info("create propagation directory")
                os.makedirs(name)

        @DoitAction
        def action_copy_initial_wave_function(targets: List[str]):
            del targets
            LOGGER.info("copy initial wave function")
            shutil.copy2(path_wfn, path_wfn_tmp)

        @DoitAction
        def action_copy_hamiltonian(targets: List[str]):
            del targets
            LOGGER.info("copy Hamiltonian")
            shutil.copy2(path_opr, path_opr_tmp)

        @DoitAction
        def action_run(targets: List[str]):
            del targets
            with cwd.WorkingDir(name):
                LOGGER.info("propagate wave function")
                cmd = ["qdtk_propagate.x"] + flag_list
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                LOGGER.info("command: %s", " ".join(cmd))
                subprocess.run(cmd, env=env)

        @DoitAction
        def action_remove_hamiltonian(targets: List[str]):
            del targets
            LOGGER.info("remove Hamiltonian")
            os.remove(path_opr_tmp)

        @DoitAction
        def action_remove_initial_wave_function(targets: List[str]):
            del targets
            LOGGER.info("remove initial wave function")
            os.remove(path_wfn_tmp)

        @DoitAction
        def action_move_final_wave_function(targets: List[str]):
            del targets
            LOGGER.info("move final wave function %s -> %s", path_restart,
                        path_final)
            shutil.move(path_restart, path_final)

        if flags["exact_diag"]:
            targets = [path_energies, path_eigenvectors]
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
                path_output,
                path_gpop,
                path_natpop,
                path_final,
            ]
            if flags["psi"]:
                targets.append(path_psi)

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
            "name": "{}:{}:run".format(basename, name),
            "actions": actions,
            "targets": targets,
            "file_dep": [path_pickle, path_wfn, path_opr],
        }

    return [task_write_parameters, task_propagate]


def relax(name: str, wave_function: str, hamiltonian: str,
          **kwargs) -> List[Callable[[], Dict[str, Any]]]:
    kwargs["relax"] = True
    kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
    kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
    kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)
    return propagate(name, wave_function, hamiltonian, **kwargs)


def improved_relax(name: str,
                   wave_function: str,
                   hamiltonian: str,
                   eig_index: int,
                   **kwargs) -> List[Callable[[], Dict[str, Any]]]:
    kwargs["improved_relax"] = True
    kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
    kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
    kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)
    kwargs["nstep_diag"] = kwargs.get("nstep_diag", 50)
    kwargs["stat_steps"] = kwargs.get("stat_steps", 80)
    kwargs["eig_index"] = eig_index
    return propagate(name, wave_function, hamiltonian, **kwargs)


def diagonalize(name: str,
                wave_function: str,
                hamiltonian: str,
                states: int,
                **kwargs) -> List[Callable[[], Dict[str, Any]]]:
    kwargs["exact_diag"] = True
    kwargs["eig_tot"] = states
    task = propagate(name, wave_function, hamiltonian, **kwargs)

    return task
