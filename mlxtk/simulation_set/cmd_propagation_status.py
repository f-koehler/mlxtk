import argparse

from mlxtk.cwd import WorkingDir
from mlxtk.simulation_set.base import SimulationSetBase


def cmd_propagation_status(self: SimulationSetBase, args: argparse.Namespace):
    total = 0.0
    self.logger.info("check propagation status of propagation: %s", args.name)
    with WorkingDir(self.working_dir):
        for simulation in self.simulations:
            progress = simulation.check_propagation_status(args.name)
            total += progress
            self.logger.info("sim %s: %6.2f%%", simulation.name, progress * 100.0)
    total = total / len(self.simulations)
    self.logger.info("total: %6.2f%%", total * 100.0)
