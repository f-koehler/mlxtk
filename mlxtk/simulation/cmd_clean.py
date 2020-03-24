import argparse
from pathlib import Path

from mlxtk.cwd import WorkingDir
from mlxtk.doit_compat import run_doit
from mlxtk.lock import LockFile
from mlxtk.simulation.base import SimulationBase


def cmd_clean(self: SimulationBase, args: argparse.Namespace):
    self.create_working_dir()
    with WorkingDir(self.working_dir):
        with LockFile(Path("run.lock")):
            run_doit(
                self.tasks_clean,
                [
                    "--process=" + str(args.jobs),
                    "--backend=json",
                    "--db-file=doit.json",
                ],
            )
