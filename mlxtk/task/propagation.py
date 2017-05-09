from mlxtk.task.expectation_value import ExpectationValue
from mlxtk.task.plot import EnergyPlot, ExpectationValuePlot, NormPlot, OverlapPlot
from mlxtk.task.task import Task
import mlxtk.log as log

import os.path
import re
import subprocess
import sys
import tqdm


class Propagation(Task):

    def __init__(self, initial_wavefunction, final_wavefunction, operator,
                 **kwargs):
        """Create a propagation task

        Args:
            initial_wavefunction (str): name of the wave function to propagate
            final_wavefunction (str): name of the resulting wave function
            operator: Hamiltonian that drives the propagation

        Keyword Args:
            dt (float): time step for the integrator
            tfinal (float): final time of the integration
            reset_time (bool): flag to reset the starting time when propagating
        """
        Task.__init__(self)

        self.name = "propagate_{}_to_{}".format(initial_wavefunction,
                                                final_wavefunction)

        self.input_files = [
            os.path.join("operators", operator + ".op"), os.path.join(
                "wavefunctions", initial_wavefunction + ".wfn")
        ]

        self.output_files = ["restart", "output", "natpop", "gpop"]

        self.symlinks = [("restart", os.path.join(
            "wavefunctions",
            final_wavefunction + ".wfn")), ("output", os.path.join(
                "outputs", self.name + ".out")), ("natpop", os.path.join(
                    "natural_populations", self.name + ".natpop")),
                         ("gpop", os.path.join("gpops", self.name + ".gpop"))]

        self.type = "Propagation"

        self.initial_wavefunction = initial_wavefunction
        self.operator = operator

        self.dt = kwargs.get("dt", 1.)
        self.tfinal = kwargs.get("tfinal", self.dt)
        self.reset_time = kwargs.get("reset_time", True)

        self.add_default_plots()

    def add_expectation_value(self, operator):
        self.write_psi = True
        self.subtasks.append(
            ExpectationValue(operator, self.initial_wavefunction))
        self.subtasks[-1].parent_task = self

        self.subtasks.append(ExpectationValuePlot(operator))
        self.subtasks[-1].parent_task = self

    def add_default_plots(self):
        self.subtasks.append(EnergyPlot())
        self.subtasks[-1].parent_task = self

        self.subtasks.append(NormPlot())
        self.subtasks[-1].parent_task = self

        self.subtasks.append(OverlapPlot())
        self.subtasks[-1].parent_task = self

    def execute(self):
        log.info("Execute \"qdtk_propagate.x\" in propagation mode")

        cmd = [
            "qdtk_propagate.x", "-rst", self.initial_wavefunction + ".wfn",
            "-opr", self.operator + ".op", "-gramschmidt", "-dt", str(self.dt),
            "-tfinal", str(self.tfinal)
        ]
        if self.reset_time:
            cmd.append("-rstzero")

        if self.write_psi:
            cmd.append("-psi")

        log.debug("Full command: %s", " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            universal_newlines=True)

        self._watch_output(process)

    def _watch_output(self, process):
        re_output = re.compile(
            r"^\s+time:\s+(\d+\.\d+(?:[eE][+-]\d+)?)\s+done$")

        # print live info about run
        with tqdm.tqdm(total=self.tfinal + self.dt, leave=False) as bar:
            for line in iter(process.stdout.readline, ""):
                m = re_output.match(line)
                if not m:
                    continue
                bar.update(self.dt)

        process.wait()
        retval = process.returncode

        if retval:
            raise subprocess.CalledProcessError
