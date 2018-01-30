import json
import os
import pickle
import shutil
import subprocess
import sys

import h5py

from mlxtk.task import task
from mlxtk.process import watch_process

from mlxtk.inout.gpop import add_gpop_to_hdf5
from mlxtk.inout.natpop import add_natpop_to_hdf5
from mlxtk.inout.output import add_output_to_hdf5


class PropagationTask(task.Task):
    def __init__(self, name, initial_wave_function, operator, **kwargs):
        self.propagation_name = name
        self.initial_wave_function = initial_wave_function
        self.operator = operator

        self.dt = kwargs.get("dt", 0.1)
        self.tfinal = kwargs.get("tfinal", 1.)
        self.psi = kwargs.get("psi", False)
        self.relax = kwargs.get("relax", False)
        self.improved_relax = kwargs.get("improved_relax", False)
        self.cont = kwargs.get("cont", False)
        self.rstzero = kwargs.get("rstzero", True)
        self.transMat = kwargs.get("transMat", False)
        self.MBop_apply = kwargs.get("MBop_apply", False)
        self.itg = kwargs.get("itg", "dp5")
        self.zvode_mf = kwargs.get("zvode_mf", 10)
        self.atol = kwargs.get("atol", 1e-10)
        self.rtol = kwargs.get("rtol", 1e-10)
        self.reg = kwargs.get("reg", 1e-8)
        self.exproj = kwargs.get("exproj", False)
        self.resetnorm = kwargs.get("resetnorm", False)
        self.gramschmidt = kwargs.get("gramschmidt", False)
        self.statsteps = kwargs.get("statsteps", 0)
        self.stat_energ_tol = kwargs.get("stat_energy_tol", 1e-10)
        self.stat_npop_tol = kwargs.get("stat_npop_tol", 1e-10)

        if self.relax:
            kwargs["task_type"] = "RelaxationTask"
            name = "relax_" + name
            if self.statsteps == 0:
                self.statsteps = 100
        elif self.improved_relax:
            name = "improved_relax_" + name
            kwargs["task_type"] = "ImprovedRelaxationTask"
        else:
            name = "propagate_" + name
            kwargs["task_type"] = "PropagationTask"

        inp_parameters = task.FileInput("parameters",
                                        os.path.join(self.propagation_name,
                                                     "parameters.pickle"))
        inp_wave_function = task.FileInput(
            "initial_wave_function",
            self.initial_wave_function + ".wave_function")
        inp_operator = task.FileInput("hamiltonian",
                                      self.operator + ".operator")

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

    def get_parameter_dict(self):
        return {
            "dt": self.dt,
            "tfinal": self.tfinal,
            "psi": self.psi,
            "relax": self.relax,
            "improved_relax": self.improved_relax,
            "rstzero": self.rstzero,
            "transMat": self.transMat,
            "MBop_apply": self.MBop_apply,
            "itg": self.itg,
            "zvode_mf": self.zvode_mf,
            "atol": self.atol,
            "rtol": self.rtol,
            "reg": self.reg,
            "exproj": self.exproj,
            "resetnorm": self.resetnorm,
            "gramschmidt": self.gramschmidt,
            "statsteps": self.statsteps,
            "stat_energ_tol": self.stat_energ_tol,
            "stat_npop_tol": self.stat_npop_tol
        }

    def get_command(self):
        parameters = self.get_parameter_dict()
        parameters["cont"] = self.cont

        if "QDTK_PREFIX" in os.environ:
            program_path = os.path.join(os.environ["QDTK_PREFIX"], "bin",
                                        "qdtk_propagate.x")
            if not os.path.exists(program_path):
                raise RuntimeError(
                    "QDTK executable \"{}\" not found".format(program_path))
        else:
            program_path = "qdtk_propagate.x"

        cmd = [program_path]
        for name in parameters:
            if isinstance(parameters[name], bool):
                if parameters[name]:
                    cmd += ["-" + name]
            else:
                cmd += ["-" + name, str(parameters[name])]

        cmd += [
            "-opr", "hamiltonian.operator", "-rst", "initial.wave_function",
            "-timing"
        ]
        return cmd

    def run_propagation(self):
        self.logger.info("copy initial wave function")
        shutil.copy2(self.initial_wave_function + ".wave_function",
                     os.path.join(self.propagation_name,
                                  "initial.wave_function"))

        self.logger.info("copy operator")
        shutil.copy2(self.operator + ".operator",
                     os.path.join(self.propagation_name,
                                  "hamiltonian.operator"))

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
        dst = os.path.join(self.propagation_name, "final.wave_function")
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
            json.dump(self.get_parameter_dict(), fhandle)

        with open(pickle_file, "wb") as fhandle:
            pickle.dump(self.get_parameter_dict(), fhandle)

    def create_hdf5(self, group=None):
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
