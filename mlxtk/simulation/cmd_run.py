import argparse
from pathlib import Path

from ..cwd import WorkingDir
from ..doit_compat import run_doit
from ..lock import LockFile


def cmd_run(self, args: argparse.Namespace):
    self.create_working_dir()
    with WorkingDir(self.working_dir):
        with LockFile(Path("run.lock")):
            run_doit(
                self.tasks_run,
                [
                    "--process=" + str(args.jobs),
                    "--backend=json",
                    "--db-file=doit.json",
                ],
            )
