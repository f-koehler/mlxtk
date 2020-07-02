import argparse
import re

from mlxtk.cwd import WorkingDir
from mlxtk.doit_compat import run_doit
from mlxtk.simulation.base import SimulationBase


def cmd_graph(self: SimulationBase, args: argparse.Namespace):
    del args

    regex = re.compile(r"^\s+\".+\"\s*->\s*\".+\";$")

    self.create_working_dir()
    with WorkingDir(self.working_dir):
        run_doit(
            self.tasks_run, ["graph", "--backend=json", "--db-file=doit.json"],
        )

        with open("tasks.dot") as fptr:
            code = fptr.readlines()

        for i, _ in enumerate(code):
            if not regex.match(code[i]):
                continue

            code[i] = code[i].replace(":", "\\n")

        with open("tasks.dot", "w") as fptr:
            fptr.writelines(code)
