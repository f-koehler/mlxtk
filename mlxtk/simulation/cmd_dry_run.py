import argparse
from pathlib import Path

from ..cwd import WorkingDir
from ..doit_compat import run_doit
from ..lock import LockFile


def cmd_dry_run(self, args: argparse.Namespace):
    del args

    self.create_working_dir()
    with WorkingDir(self.working_dir):
        with LockFile(Path("run.lock")):
            run_doit(
                self.tasks_dry_run,
                [
                    "--backend=json",
                    "--db-file=doit.json",
                ],
            )
