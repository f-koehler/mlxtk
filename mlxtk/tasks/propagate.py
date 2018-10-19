import copy
import gzip
import os
import pickle
import subprocess
from typing import Any, Callable, Dict, List

from .. import cwd, log
from ..inout.eigenenergies import read_eigenenergies_ascii, write_eigenenergies_hdf5
from ..inout.gpop import read_gpop_ascii, write_gpop_hdf5
from ..inout.natpop import read_natpop_ascii, write_natpop_hdf5
from ..inout.output import read_output_ascii, write_output_hdf5
from ..inout.psi import read_psi_ascii, write_psi_hdf5

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


def propagate(
    name: str, wave_function: str, hamiltonian: str, **kwargs
) -> List[Callable[[], Dict[str, Any]]]:
    keep_psi = kwargs.get("keep_psi", False)
    flags, flag_list = create_flags(**kwargs)
    flag_list += ["-rst", "initial.wfn", "-opr", "hamiltonian.mb_opr"]

    path_pickle = name + ".prop_pickle"

    def task_write_parameters() -> Dict[str, Any]:
        def action_write_parameters(targets: List[str]):
            obj = [name, wave_function, hamiltonian, flags]

            with open(targets[0], "wb") as fp:
                pickle.dump(obj, fp)

        return {
            "name": "propagation:{}:write_parameters".format(name),
            "actions": [action_write_parameters],
            "targets": [path_pickle],
        }

    def task_propagate() -> Dict[str, Any]:
        path_output = os.path.join(name, "output")
        path_gpop = os.path.join(name, "gpop")
        path_natpop = os.path.join(name, "natpop")
        path_final = os.path.join(name, "restart")
        path_final_gz = os.path.join(name, "final.wfn.gz")
        path_psi = os.path.join(name, "psi")
        path_energies = os.path.join(name, "eigenenergies")
        path_eigenvectors = os.path.join(name, "eigenvectors")

        path_output_hdf5 = path_output + ".hdf5"
        path_gpop_hdf5 = path_gpop + ".hdf5"
        path_natpop_hdf5 = path_natpop + ".hdf5"
        path_psi_hdf5 = os.path.join(name, "psi.hdf5")
        path_energies_hdf5 = path_energies + ".hdf5"
        path_eigenvectors_hdf5 = path_eigenvectors + ".hdf5"

        path_wfn = wave_function + ".wfn.gz"
        path_opr = hamiltonian + ".mb_opr.gz"
        path_wfn_tmp = os.path.join(name, "initial.wfn")
        path_opr_tmp = os.path.join(name, "hamiltonian.mb_opr")

        def action_create_dir(targets: List[str]):
            del targets
            if not os.path.exists(name):
                LOGGER.info("create propagation directory")
                os.makedirs(name)

        def action_copy_initial_wave_function(targets: List[str]):
            del targets
            LOGGER.info("copy initial wave function")
            with gzip.open(path_wfn, "rb") as fp_in:
                with open(path_wfn_tmp, "w") as fp_out:
                    fp_out.write(fp_in.read().decode())

        def action_copy_hamiltonian(targets: List[str]):
            del targets
            LOGGER.info("copy Hamiltonian")
            with gzip.open(path_opr, "rb") as fp_in:
                with open(path_opr_tmp, "w") as fp_out:
                    fp_out.write(fp_in.read().decode())

        def action_propagate(targets: List[str]):
            del targets
            with cwd.WorkingDir(name):
                LOGGER.info("propagate wave function")
                cmd = ["qdtk_propagate.x"] + flag_list
                env = os.environ.copy()
                env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                LOGGER.info("command: %s", " ".join(cmd))
                subprocess.run(cmd, env=env)

        def action_convert_output(targets: List[str]):
            del targets
            LOGGER.info("convert ouput to HDF5")
            write_output_hdf5(path_output_hdf5, read_output_ascii(path_output))
            os.remove(path_output)

        def action_convert_gpop(targets: List[str]):
            del targets
            LOGGER.info("convert one-body densities to HDF5")
            write_gpop_hdf5(path_gpop_hdf5, read_gpop_ascii(path_gpop))
            os.remove(path_gpop)

        def action_convert_natpop(targets: List[str]):
            del targets
            LOGGER.info("convert natural populations to HDF5")
            write_natpop_hdf5(path_natpop_hdf5, read_natpop_ascii(path_natpop))
            os.remove(path_natpop)

        def action_convert_psi(targets: List[str]):
            del targets
            LOGGER.info("convert psi file to HDF5")
            write_psi_hdf5(path_psi_hdf5, read_psi_ascii(path_psi))
            if not keep_psi:
                os.remove(path_psi)

        def action_convert_final_wfn(targets: List[str]):
            del targets
            LOGGER.info("compress final wave function")
            with gzip.open(path_final_gz, "wb") as fout:
                with open(path_final, "r") as fin:
                    fout.write(fin.read().encode())
            os.remove(path_final)

        def action_remove_hamiltonian(targets: List[str]):
            del targets
            LOGGER.info("remove Hamiltonian")
            os.remove(path_opr_tmp)

        def action_remove_initial_wave_function(targets: List[str]):
            del targets
            LOGGER.info("remove initial wave function")
            os.remove(path_wfn_tmp)

        def action_convert_eigenenergies(targets: List[str]):
            del targets
            LOGGER.info("convert eigenenergies to HDF5")
            write_eigenenergies_hdf5(
                path_energies_hdf5, read_eigenenergies_ascii(path_energies)
            )
            os.remove(path_energies)

        def action_convert_eigenvectors(targets: List[str]):
            del targets
            LOGGER.info("convert eigenstates to HDF5")
            write_psi_hdf5(path_eigenvectors_hdf5, read_psi_ascii(path_eigenvectors))
            os.remove(path_eigenvectors)

        if flags["exact_diag"]:
            targets = [path_energies_hdf5, path_eigenvectors_hdf5]
            actions = [
                action_create_dir,
                action_copy_initial_wave_function,
                action_copy_hamiltonian,
                action_propagate,
                action_remove_hamiltonian,
                action_remove_initial_wave_function,
                action_convert_eigenenergies,
                action_convert_eigenvectors,
            ]
        else:
            targets = [
                path_output_hdf5,
                path_gpop_hdf5,
                path_natpop_hdf5,
                path_final_gz,
            ]
            if flags["psi"]:
                targets.append(path_psi_hdf5)

            actions = [
                action_create_dir,
                action_copy_initial_wave_function,
                action_copy_hamiltonian,
                action_propagate,
                action_remove_hamiltonian,
                action_remove_initial_wave_function,
                action_convert_output,
                action_convert_gpop,
                action_convert_natpop,
                action_convert_final_wfn,
            ]
            if flags["psi"]:
                actions.append(action_convert_psi)

        return {
            "name": "propagation:{}:propagate".format(name),
            "actions": actions,
            "targets": targets,
            "file_dep": [path_pickle, path_wfn, path_opr],
        }

    return [task_write_parameters, task_propagate]


def relax(
    name: str, wave_function: str, hamiltonian: str, **kwargs
) -> List[Callable[[], Dict[str, Any]]]:
    kwargs["relax"] = True
    kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
    kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
    kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)
    return propagate(name, wave_function, hamiltonian, **kwargs)


def improved_relax(
    name: str, wave_function: str, hamiltonian: str, eig_index: int, **kwargs
) -> List[Callable[[], Dict[str, Any]]]:
    kwargs["improved_relax"] = True
    kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
    kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
    kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)
    kwargs["nstep_diag"] = kwargs.get("nstep_diag", 50)
    kwargs["stat_steps"] = kwargs.get("stat_steps", 80)
    kwargs["eig_index"] = eig_index
    return propagate(name, wave_function, hamiltonian, **kwargs)


def diagonalize(
    name: str, wave_function: str, hamiltonian: str, states: int, **kwargs
) -> List[Callable[[], Dict[str, Any]]]:
    kwargs["exact_diag"] = True
    kwargs["eig_tot"] = states
    task = propagate(name, wave_function, hamiltonian, **kwargs)

    return task
