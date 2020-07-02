import copy
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy

from mlxtk import cwd
from mlxtk.doit_compat import DoitAction
from mlxtk.hashing import hash_file
from mlxtk.inout.eigenbasis import add_eigenbasis_to_hdf5, read_eigenbasis_ascii
from mlxtk.inout.gpop import add_gpop_to_hdf5, read_gpop_ascii
from mlxtk.inout.natpop import add_natpop_to_hdf5, read_natpop_ascii
from mlxtk.inout.output import add_output_to_hdf5, read_output_ascii
from mlxtk.log import get_logger
from mlxtk.tasks.task import Task
from mlxtk.temporary_dir import TemporaryDir
from mlxtk.util import copy_file, make_path

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
    "gauge_reg": float,
    "gauge_relax": float,
    "diag_gauge_oper": str,
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

    non_qdtk_flags = ["extra_files", "gauge_diag_oper"]

    for flag in kwargs:
        if flag not in FLAG_TYPES:
            if flag not in non_qdtk_flags:
                raise RuntimeError('Unknown flag "{}"'.format(flag))
        else:
            flags[flag] = kwargs[flag]

    if flags["relax"]:
        flags["statsteps"] = flags.get("statsteps", 50)

    if flags["improved_relax"]:
        flags["statsteps"] = flags.get("statsteps", 40)
        flags["eig_index"] = flags.get("eig_index", 1)
        flags["nstep_diag"] = flags.get("nstep_diag", 30)

    if flags["itg"] == "zvode":
        flags["zvode_mf"] = flags.get("zvode_mf", 10)

    flag_list = []
    for flag in flags:
        if FLAG_TYPES[flag] == bool:
            if flags[flag]:
                flag_list.append("-" + flag)
        else:
            flag_list += ["-" + flag, str(flags[flag])]

    return flags, flag_list


class Propagate(Task):
    def __init__(
        self,
        name: str,
        wave_function: Union[str, Path],
        hamiltonian: Union[str, Path],
        **kwargs,
    ):
        self.name = name
        self.wave_function = wave_function
        self.hamiltonian = hamiltonian
        self.logger = get_logger(__name__ + ".Propagate")
        self.diag_gauge_oper: Optional[str] = kwargs.get("diag_gauge_oper", None)

        self.flags, self.flag_list = create_flags(**kwargs)
        self.flag_list += ["-rst", "restart", "-opr", "hamiltonian"]

        if self.flags.get("gauge", "standard") == "diagonalization":
            self.flag_list += ["-diag_gauge_oper", "gauge_oper"]

        if self.flags["relax"]:
            self.basename = "relaxation"
        elif self.flags["improved_relax"]:
            self.basename = "improved_relaxation"
        elif self.flags["exact_diag"]:
            self.basename = "exact_diagonalization"
        else:
            self.basename = "propagation"

        # compute required paths
        self.path_name = Path(self.name)
        self.path_pickle = Path(self.name + ".prop_pickle")
        self.path_wave_function = make_path(wave_function).with_suffix(".wfn")
        self.path_hamiltonian = make_path(hamiltonian).with_suffix(".mb_opr")
        self.path_diag_gauge_oper: Optional[Path] = Path(
            self.diag_gauge_oper
        ) if self.diag_gauge_oper else None

        self.path_hdf5 = self.path_name / (
            "exact_diag.h5" if self.flags["exact_diag"] else "propagate.h5"
        )

        if self.flags["exact_diag"]:
            self.qdtk_files: List[str] = []
        else:
            self.qdtk_files = ["final.wfn"]
            if self.flags["psi"]:
                self.qdtk_files.append("psi")

        self.qdtk_files += kwargs.get("extra_files", [])

    def task_write_parameters(self) -> Dict[str, Any]:
        @DoitAction
        def action_write_parameters(targets: List[str]):
            del targets

            obj = [
                self.name,
                str(self.wave_function),
                str(self.hamiltonian),
                self.flags,
            ]
            if self.diag_gauge_oper:
                obj.append(self.diag_gauge_oper)

            path_temp = self.path_name.resolve().with_name("." + self.path_name.name)
            if path_temp.exists() and self.flags["exact_diag"]:
                shutil.rmtree(path_temp)
            if path_temp.exists():
                if self.path_pickle.exists():
                    with open(self.path_pickle, "rb") as fptr:
                        if pickle.load(fptr) != obj:
                            self.logger.warn("propagation parameters changed")
                            shutil.rmtree(path_temp)
                else:
                    shutil.rmtree(path_temp)

            with open(self.path_pickle, "wb") as fp:
                pickle.dump(obj, fp, protocol=3)

        return {
            "name": "{}:{}:write_parameters".format(self.basename, self.name),
            "actions": [action_write_parameters],
            "targets": [self.path_pickle],
        }

    def task_propagate(self) -> Dict[str, Any]:
        @DoitAction
        def action_run(targets: List[str]):
            del targets

            if not self.path_name.exists():
                self.path_name.mkdir()

            path_temp = self.path_name.resolve().with_name("." + self.path_name.name)

            if path_temp.exists():
                path_temp_operator = path_temp / "hamiltonian"
                if not path_temp_operator.exists():
                    self.logger.debug(
                        'temporary operator "%s" does not exist',
                        str(path_temp_operator),
                    )
                    shutil.rmtree(path_temp)
                else:
                    hash_a = hash_file(path_temp_operator)
                    hash_b = hash_file(self.path_hamiltonian.resolve())
                    if hash_a != hash_b:
                        self.logger.debug("operator hashes do not match")
                        shutil.rmtree(path_temp)

            if path_temp.exists():
                path_temp_wfn = path_temp / "initial"
                if not path_temp_wfn.exists():
                    self.logger.debug(
                        'temporary wave function "%s" does not exist',
                        str(path_temp_wfn),
                    )
                    shutil.rmtree(path_temp)
                else:
                    hash_a = hash_file(path_temp_wfn)
                    hash_b = hash_file(self.path_wave_function.resolve())
                    if hash_a != hash_b:
                        self.logger.debug("wave function hashes do not match")
                        shutil.rmtree(path_temp)

            if path_temp.exists():
                self.logger.info("attempt continuation run")
                self.flags["cont"] = True
                self.flags["rstzero"] = False
                self.flag_list.append("-cont")
                try:
                    self.flag_list.remove("-rstzero")
                except ValueError:
                    pass

            with TemporaryDir(path_temp) as tmpdir:
                operator = self.path_hamiltonian.resolve()
                wave_function = self.path_wave_function.resolve()
                output_dir = self.path_name.resolve()
                hdf5_file = self.path_hdf5.resolve()
                diag_gauge_oper = (
                    self.path_diag_gauge_oper.resolve()
                    if self.path_diag_gauge_oper
                    else None
                )

                with cwd.WorkingDir(tmpdir.path):
                    self.logger.info("propagate wave function")
                    if not self.flags["cont"]:
                        copy_file(operator, "hamiltonian")
                        copy_file(wave_function, "initial")
                        copy_file(wave_function, "restart")
                    if self.flags.get("gauge", "standard") == "diagonalization":
                        if not diag_gauge_oper:
                            raise ValueError(
                                "no operator specified for diagonalization gauge"
                            )
                        copy_file(diag_gauge_oper, "gauge_oper")
                    cmd = ["qdtk_propagate.x"] + self.flag_list
                    env = os.environ.copy()
                    env["OMP_NUM_THREADS"] = env.get("OMP_NUM_THREADS", "1")
                    self.logger.info("command: %s", " ".join(cmd))
                    result = subprocess.run(cmd, env=env)
                    if result.returncode != 0:
                        raise RuntimeError("Failed to run qdtk_propagate.x")

                    if self.flags["exact_diag"]:
                        with h5py.File("result.h5", "w") as fptr:
                            add_eigenbasis_to_hdf5(fptr, *read_eigenbasis_ascii("."))
                    else:
                        shutil.move("restart", "final.wfn")
                        with h5py.File("result.h5", "w") as fptr:
                            add_gpop_to_hdf5(
                                fptr.create_group("gpop"), *read_gpop_ascii("gpop")
                            )
                            add_natpop_to_hdf5(
                                fptr.create_group("natpop"),
                                *read_natpop_ascii("natpop"),
                            )
                            add_output_to_hdf5(
                                fptr.create_group("output"),
                                *read_output_ascii("output"),
                            )

                            if "gauge" in self.flags:
                                if self.flags["gauge"] != "standard":
                                    time, error = numpy.loadtxt(
                                        "constraint_error.txt", unpack=True
                                    )
                                    group = fptr.create_group("constraint_error")
                                    group.create_dataset(
                                        "time", time.shape, dtype=time.dtype
                                    )[:] = time
                                    group.create_dataset(
                                        "error", error.shape, dtype=error.dtype
                                    )[:] = error

                    shutil.move("result.h5", hdf5_file)

                    for fname in self.qdtk_files:
                        copy_file(fname, output_dir / fname)

                    tmpdir.complete = True

        deps = [self.path_pickle, self.path_wave_function, self.path_hamiltonian]
        if self.path_diag_gauge_oper:
            deps.append(self.path_diag_gauge_oper)

        return {
            "name": "{}:{}:run".format(self.basename, self.name),
            "actions": [action_run],
            "targets": [str(self.path_name / fname) for fname in self.qdtk_files]
            + [str(self.path_hdf5)],
            "file_dep": deps,
            "verbosity": 2,
        }

    def get_tasks_run(self) -> List[Callable[[], Dict[str, Any]]]:
        return [self.task_write_parameters, self.task_propagate]


class Relax(Propagate):
    def __init__(
        self,
        name: str,
        wave_function: Union[str, Path],
        hamiltonian: Union[str, Path],
        **kwargs,
    ):
        kwargs["relax"] = True
        kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
        kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
        kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)

        super().__init__(name, wave_function, hamiltonian, **kwargs)

        self.logger = get_logger(__name__ + ".Relax")


class ImprovedRelax(Propagate):
    def __init__(
        self,
        name: str,
        wave_function: Union[str, Path],
        hamiltonian: Union[str, Path],
        eig_index: int,
        **kwargs,
    ):
        kwargs["improved_relax"] = True
        kwargs["tfinal"] = kwargs.get("tfinal", 1000.0)
        kwargs["stat_energ_tol"] = kwargs.get("stat_energ_tol", 1e-8)
        kwargs["stat_npop_tol"] = kwargs.get("stat_npop_tol", 1e-6)
        kwargs["nstep_diag"] = kwargs.get("nstep_diag", 50)
        kwargs["statsteps"] = kwargs.get("stat_steps", 80)
        kwargs["eig_index"] = eig_index

        super().__init__(name, wave_function, hamiltonian, **kwargs)

        self.logger = get_logger(__name__ + ".ImprovedRelax")


class Diagonalize(Propagate):
    def __init__(
        self,
        name: str,
        wave_function: Union[str, Path],
        hamiltonian: Union[str, Path],
        number_of_states: int,
        **kwargs,
    ):
        kwargs["exact_diag"] = True
        kwargs["eig_tot"] = number_of_states

        super().__init__(name, wave_function, hamiltonian, **kwargs)

        self.logger = get_logger(__name__ + ".Diagonalize")
