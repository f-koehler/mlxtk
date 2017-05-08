from mlxtk.task.task import Task
from mlxtk.task.propagation import Propagation
import mlxtk.log as log

import subprocess


class Relaxation(Propagation):

    def __init__(self, initial_wavefunction, final_wavefunction, operator,
                 **kwargs):
        Propagation.__init__(self, initial_wavefunction, final_wavefunction,
                             operator, **kwargs)

        self.name = "relax_{}_to_{}".format(initial_wavefunction,
                                                final_wavefunction)
        self.type = "Relaxation"

        self.statsteps = kwargs.get("statsteps", 0)

    def execute(self):
        log.info("Execute \"qdtk_propagate.x\" in relaxation mode")

        cmd = [
            "qdtk_propagate.x", "-relax", "-rst",
            self.initial_wavefunction + ".wfn", "-opr", self.operator + ".op",
            "-gramschmidt", "-dt", str(self.dt), "-tfinal", str(self.tfinal),
            "-stateps", str(self.statsteps)
        ]
        log.debug("Full command: %s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)

        self._watch_output(process)
