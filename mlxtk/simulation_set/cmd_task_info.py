import argparse

from mlxtk.cwd import WorkingDir
from mlxtk.simulation_set.base import SimulationSetBase


def cmd_task_info(self: SimulationSetBase, args: argparse.Namespace):
    self.create_working_dir()

    with WorkingDir(self.working_dir):
        self.simulations[args.index].main(["task-info", args.name])
