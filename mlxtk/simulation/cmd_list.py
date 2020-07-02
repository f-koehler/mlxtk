import argparse

from mlxtk.cwd import WorkingDir
from mlxtk.doit_compat import run_doit
from mlxtk.simulation.base import SimulationBase


def cmd_list(self: SimulationBase, args: argparse.Namespace):
    del args

    self.create_working_dir()
    with WorkingDir(self.working_dir):
        run_doit(
            self.tasks_run, ["list", "--backend=json", "--db-file=doit.json"],
        )
