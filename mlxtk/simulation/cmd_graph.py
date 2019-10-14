import argparse
import re

from ..cwd import WorkingDir
from ..doit_compat import run_doit


def cmd_graph(self, args: argparse.Namespace):
    del args

    regex = re.compile(r"^\s+\".+\"\s*->\s*\".+\";$")

    self.create_working_dir()
    with WorkingDir(self.working_dir):
        run_doit(
            self.tasks_run,
            ["graph", "--backend=json", "--db-file=doit.json"],
        )

        with open("tasks.dot") as fptr:
            code = fptr.readlines()

        for i, _ in enumerate(code):
            if not regex.match(code[i]):
                continue

            code[i] = code[i].replace(":", "\\n")

        with open("tasks.dot", "w") as fptr:
            fptr.writelines(code)
