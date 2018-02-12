import os
import shutil
import sys

import h5py

from mlxtk.qdtk_executable import find_qdtk_executable
from mlxtk.inout import hdf5
from mlxtk.task import task
from mlxtk.process import watch_process
from mlxtk.inout.expval import add_expval_to_hdf5


class ExpectationValueTask(task.Task):
    def __init__(self, propagation, operator, **kwargs):
        kwargs["task_type"] = "ExpectationValueTask"

        self.operator = operator
        self.propagation = propagation

        inp_wave_function = task.FileInput(
            "initial_wave_function",
            propagation.initial_wave_function + ".wave_function")
        inp_operator = task.FileInput("operator",
                                      propagation.operator + ".operator")
        inp_psi_file = task.FileInput("psi_file",
                                      os.path.join(
                                          propagation.propagation_name, "psi"))

        out_expval = task.FileOutput("expval_" + operator,
                                     os.path.join(propagation.propagation_name,
                                                  operator + ".expval"))

        task.Task.__init__(
            self,
            propagation.propagation_name + "_expval_" + operator,
            self.compute_expectation_value,
            inputs=[inp_wave_function, inp_operator, inp_psi_file],
            outputs=[out_expval],
            **kwargs)

        if not propagation.psi:
            self.logger.warn("propagation does not seem to create a psi file")

    def get_command(self):
        program_path = find_qdtk_executable("qdtk_expect.x")
        self.logger.info("use propagation executable: " + program_path)

        return [
            program_path, "-psi", "psi", "-rst", "restart", "-opr",
            self.operator + ".operator", "-save", self.operator + ".expval"
        ]

    def compute_expectation_value(self):
        self.logger.info("copy operator")
        shutil.copy2(self.operator + ".operator",
                     os.path.join(self.propagation.propagation_name,
                                  self.operator + ".operator"))

        self.logger.info("run qdtk_expect.x")
        command = self.get_command()
        self.logger.debug("command: %s", " ".join(command))
        working_dir = self.propagation.propagation_name
        self.logger.debug("working directory: %s", working_dir)

        watch_process(
            command,
            self.logger.info,
            self.logger.warn,
            cwd=working_dir,
            stdout=sys.stdout,
            stderr=sys.stderr)

    def create_hdf5(self, group=None):
        if self.is_running():
            raise hdf5.IncompleteHDF5(
                "Possibly running ExpectationValueTask, no data can be added")

        opened_file = group is None
        if opened_file:
            self.logger.info("create new hdf5 file")
            group = h5py.File(self.propagation.propagation_name + ".hdf5", "w")
        else:
            if self.propagation.propagation_name not in group:
                group = group.create_group(self.propagation.propagation_name)
            else:
                group = group[self.propagation.propagation_name]

        expval = os.path.join(self.propagation.propagation_name,
                              self.operator + ".expval")
        if not os.path.exists(expval):
            raise RuntimeError(
                "Expectation value \"{}\" does not exist".format(expval))
        add_expval_to_hdf5(group, expval)

        if opened_file:
            group.close()
