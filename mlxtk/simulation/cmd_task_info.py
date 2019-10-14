import argparse

from ..cwd import WorkingDir
from ..doit_compat import run_doit
from .base import SimulationBase


def cmd_task_info(self: SimulationBase, args: argparse.Namespace):
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
