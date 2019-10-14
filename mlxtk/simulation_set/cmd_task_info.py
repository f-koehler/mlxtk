import argparse

from ..cwd import WorkingDir


def cmd_task_info(self, args: argparse.Namespace):
    self.create_working_dir()

    with WorkingDir(self.working_dir):
        self.simulations[args.index].main(["task-info", args.name])
