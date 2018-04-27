import json
import os
import pickle
import shutil
import subprocess
import sys

import h5py

from mlxtk.task import task
from mlxtk.process import watch_process
from mlxtk.qdtk_executable import find_qdtk_executable

from mlxtk.inout import hdf5
from mlxtk.inout.gpop import add_gpop_to_hdf5
from mlxtk.inout.natpop import add_natpop_to_hdf5
from mlxtk.inout.output import add_output_to_hdf5

FLAG_TYPES = {
    "rst": str,
    "opr": str,
    "dt": float,
    "tfinal": float,
    "psi": bool,
    "relax": bool,
    "improved_relax": bool,
    "cont": bool,
    "rstzero": bool,
    "transMat": bool,
    "MBop_apply": bool,
    "itg": str,
    "zvode_mf": int,
    "atol": float,
    "rtol": float,
    "reg": float,
    "exproj": bool,
    "resetnorm": bool,
    "gramschmidt": bool,
    "statsteps": int,
    "stat_energy_tol": float,
    "stat_npop_tol": float,
    "gauge": str,
    "pruning_method": str,
    "timing": bool
}

DEFAULT_FLAGS = {
    "rst": "initial.wfn",
    "opr": "hamiltonian.opr",
    "dt": 0.1,
    "tfinal": 1.,
    "psi": False,
    "relax": False,
    "improved_relax": False,
    "cont": False,
    "rstzero": False,
    "transMat": False,
    "MBop_apply": False,
    "itg": "dp5",
    "atol": 1e-12,
    "rtol": 1e-12,
    "reg": 1e-8,
    "exproj": True,
    "resetnorm": False,
    "gramschmidt": False,
    "timing": True
}


def create_flags(**kwargs):
    flags = DEFAULT_FLAGS

    for flag in FLAG_TYPES:
        if flag not in kwargs:
            continue
        flags[flag] = kwargs[flag]

    if flags["improved_relax"]:
        flags["statsteps"] = flags.get("statsteps", 100)

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


class PropagationTask(task.Task):
    def __init__(self, name, initial_wave_function, operator, **kwargs):
        self.propagation_name = name

        self.initial_wave_function = initial_wave_function + ".wfn"
        self.operator = operator + ".opr"

        self.flags, self.flag_list = create_flags(**kwargs)

        if self.flags["relax"]:
            kwargs["task_type"] = "RelaxationTask"
            name = "relax_" + name
        elif self.flags["improved_relax"]:
            name = "improved_relax_" + name
            kwargs["task_type"] = "ImprovedRelaxationTask"
        else:
            name = "propagate_" + name
            kwargs["task_type"] = "PropagationTask"

        inp_parameters = task.FileInput("parameters",
                                        os.path.join(self.propagation_name,
                                                     "parameters.pickle"))
        inp_wave_function = task.FileInput("initial_wave_function",
                                           self.flags["rst"])
        inp_operator = task.FileInput("hamiltonian", self.flags["opr"])

        out_restart = task.FileOutput("final_wave_function",
                                      os.path.join(self.propagation_name,
                                                   "restart"))
        out_gpop = task.FileOutput("global_population",
                                   os.path.join(self.propagation_name, "gpop"))
        out_npop = task.FileOutput("natural_population",
                                   os.path.join(self.propagation_name,
                                                "natpop"))
        out_output = task.FileOutput("qdtk_output",
                                     os.path.join(self.propagation_name,
                                                  "output"))

        task.Task.__init__(
            self,
            name,
            self.run_propagation,
            inputs=[inp_parameters, inp_wave_function, inp_operator],
            outputs=[out_restart, out_gpop, out_npop, out_output],
            **kwargs)

        self.preprocess_steps.append(self.write_propagation_parameters)

    def get_final_wave_function_name(self):
        return os.path.join(self.propagation_name, "final")

    def get_command(self):
        program_path = find_qdtk_executable("qdtk_propagate.x")
        self.logger.info("use propagation executable: " + program_path)

        cmd = [program_path] + self.flag_list
        return cmd

    def run_propagation(self):
        self.logger.info("copy initial wave function")
        shutil.copy2(self.initial_wave_function,
                     os.path.join(self.propagation_name, "initial.wfn"))

        self.logger.info("copy operator")
        shutil.copy2(self.operator,
                     os.path.join(self.propagation_name, "hamiltonian.opr"))

        self.logger.info("run qdtk_propagate.x")
        command = self.get_command()
        self.logger.debug("command: %s", " ".join(command))
        working_dir = self.propagation_name
        self.logger.debug("working directory: %s", working_dir)

        watch_process(
            command,
            self.logger.info,
            self.logger.warn,
            cwd=working_dir,
            stdout=sys.stdout,
            stderr=sys.stderr)

        self.logger.info("create symlink to final wave function")
        src = os.path.join(self.propagation_name, "restart")
        dst = os.path.join(self.propagation_name, "final.wfn")
        self.logger.debug("%s -> %s", src, dst)

        subprocess.check_output(["ln", "-srf", src, dst])

    def write_propagation_parameters(self):
        """Write the propagation parameters to disk

        The propagation parameters are writen to a human-readable JSON file
        (``parameters.json``) and to a machine-readable pickle file
        (``parameters.pickle``) within the task directory
        """
        json_file = os.path.join(self.propagation_name, "parameters.json")
        pickle_file = os.path.join(self.propagation_name, "parameters.pickle")
        with open(json_file, "w") as fhandle:
            json.dump(self.flags, fhandle)

        with open(pickle_file, "wb") as fhandle:
            pickle.dump(self.flags, fhandle)

    def create_hdf5(self, group=None):
        if self.is_running():
            raise hdf5.IncompleteHDF5(
                "Possibly running ExpectationValueTask, no data can be added")

        opened_file = group is None
        if opened_file:
            self.logger.info("create new hdf5 file")
            group = h5py.File(self.propagation_name + ".hdf5", "w")
        else:
            group = group.create_group(self.propagation_name)

        gpop = os.path.join(self.propagation_name, "gpop")
        if os.path.exists(gpop):
            add_gpop_to_hdf5(group, gpop)
        else:
            self.logger.warn("gpop file does not exist, incomplete dataset")

        natpop = os.path.join(self.propagation_name, "natpop")
        if os.path.exists(natpop):
            add_natpop_to_hdf5(group, natpop)
        else:
            self.logger.warn("natpop file does not exist, incomplete dataset")

        output = os.path.join(self.propagation_name, "output")
        if os.path.exists(natpop):
            add_output_to_hdf5(group, output)
        else:
            self.logger.warn("output file does not exist, incomplete dataset")

        if opened_file:
            group.close()
