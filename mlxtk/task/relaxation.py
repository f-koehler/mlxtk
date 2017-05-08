from mlxtk.task.task import Task
from mlxtk.task.propagation import Propagation

import logging
import subprocess


class Relaxation(Propagation):

    def __init__(self, initial_wavefunction, final_wavefunction, operator,
                 **kwargs):
        Propagation.__init__(self, initial_wavefunction, final_wavefunction,
                             operator, **kwargs)
        Task.__init__(self, "relax_{}_to_{}".format(initial_wavefunction,
                                                    final_wavefunction))

        self.type = "Relaxation"

        self.statsteps = kwargs.get("statsteps", 0)

    def _execute(self):
        logging.info("Execute relaxation procedure: %s -> %s",
                     self.initial_wavefunction, self.final_wavefunction)
        cmd = [
            "qdtk_propagate.x", "-relax", "-rst",
            self.initial_wavefunction + ".wfn", "-opr", self.operator + ".op",
            "-gramschmidt", "-dt", str(self.dt), "-tfinal", str(self.tfinal),
            "-stateps", str(self.statsteps)
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=self.get_working_dir(),
            universal_newlines=True)

        return process
