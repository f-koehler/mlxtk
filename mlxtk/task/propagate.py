import json
import os
import shutil
import subprocess
import sys

from mlxtk.task import task


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
        self.rstzero = kwargs.get("rstzero", False)
        self.transMat = kwargs.get("transMat", False)
        self.MBop_apply = kwargs.get("MBop_apply", False)
        self.itg = kwargs.get("itg", "zvode")
        self.zvode_mf = kwargs.get("zvode_mf", 10)
        self.atol = kwargs.get("atol", 1e-10)
        self.rtol = kwargs.get("rtol", 1e-10)
        self.reg = kwargs.get("reg", 1e-12)
        self.exproj = kwargs.get("exproj", False)
        self.resetnorm = kwargs.get("resetnorm", False)
        self.gramschmidt = kwargs.get("gramschmidt", False)
        self.statsteps = kwargs.get("statsteps", 0)
        self.stat_energ_tol = kwargs.get("stat_energy_tol", 1e-9)
        self.stat_npop_tol = kwargs.get("stat_npop_tol", 1e-9)

        if self.relax:
            kwargs["task_type"] = "RelaxationTask"
            name = "relax_" + name
            if self.statsteps == 0:
                self.statsteps = 30
        elif self.improved_relax:
            name = "improved_relax_" + name
            kwargs["task_type"] = "ImprovedRelaxationTask"
        else:
            name = "propagate_" + name
            kwargs["task_type"] = "PropagationTask"

        inp_parameters = task.FileInput("parameters",
                                        os.path.join(self.propagation_name,
                                                     "parameters.json"))
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
                                   os.path.join(self.propagation_name, "npop"))
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
            "cont": self.cont,
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
        cmd = ["qdtk_propagate.x"]
        for name in parameters:
            if isinstance(parameters[name], bool):
                if parameters[name]:
                    cmd += ["-" + name]
            else:
                cmd += ["-" + name, str(parameters[name])]

        cmd += [
            "-opr", self.operator + ".operator", "-rst",
            self.initial_wave_function + ".wave_function"
        ]
        return cmd

    def run_propagation(self):
        self.logger.info("copy initial wave function")
        shutil.copy2(
            os.path.join(self.cwd,
                         self.initial_wave_function + ".wave_function"),
            os.path.join(self.cwd, self.propagation_name,
                         "initial.wave_function"))

        self.logger.info("copy operator")
        shutil.copy2(
            os.path.join(self.cwd, self.operator + ".operator"),
            os.path.join(self.cwd, self.propagation_name,
                         "hamiltonian.operator"))

        self.logger.info("run qdtk_propagate.x")
        command = self.get_command()
        self.logger.debug("command: %s", " ".join(command))
        working_dir = os.path.join(self.cwd, self.propagation_name)
        self.logger.debug("working directory: %s", working_dir)
        process = subprocess.Popen(
            command, cwd=working_dir, stdout=sys.stdout, stderr=sys.stderr)
        if process.wait():
            raise subprocess.CalledProcessError(command, process.returncode)

    def write_propagation_parameters(self):
        with open(
                os.path.join(self.cwd, self.propagation_name,
                             self.propagation_name), "w") as fhandle:
            json.dump(self.get_parameter_dict(), fhandle)
