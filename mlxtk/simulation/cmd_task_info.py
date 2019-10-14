import argparse

from ..cwd import WorkingDir
from ..doit_compat import run_doit


def cmd_task_info(self, args: argparse.Namespace):
    with WorkingDir(self.working_dir):
        run_doit(
            self.tasks_run,
            [
                "info",
                "--backend=json",
                "--db-file=doit.json",
                args.name,
            ],
        )
