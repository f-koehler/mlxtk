from mlxtk.task.task import SubTask
import mlxtk.log as log

import os.path
import subprocess
import sys


class ExpectationValue(SubTask):

    def __init__(self, operator, wavefunction):
        SubTask.__init__(self, None)

        self.type = "ExpectationValue"
        self.name = "expectation_value_{}".format(operator)
        self.wavefunction = wavefunction

        self.operator = operator

        self.input_files = [os.path.join("operators", self.operator + ".op")]
        self.output_files = [self.operator + ".expval"]
        self.symlinks = []

    def execute(self):
        log.info("Execute \"qdtk_expect.x\"")

        cmd = [
            "qdtk_expect.x", "-rst", self.wavefunction + ".wfn", "-opr",
            self.operator + ".op", "-save", self.operator + ".expval", "-psi",
            "psi"
        ]
        log.debug("Full command: %s", " ".join(cmd))

        process = subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=sys.stderr, universal_newlines=True)

        process.wait()
        retval = process.returncode
        if retval:
            raise subprocess.CalledProcessError
