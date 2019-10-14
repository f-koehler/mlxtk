import argparse

from ..cwd import WorkingDir


def cmd_propagation_status(self, args: argparse.Namespace):
    total = 0.
    self.logger.info("check propagation status of propagation: %s", args.name)
    with WorkingDir(self.working_dir):
        for simulation in self.simulations:
            progress = simulation.check_propagation_status(args.name)
            total += progress
            self.logger.info("sim %s: %6.2f%%", simulation.name,
                             progress * 100.)
    total = total / len(self.simulations)
    self.logger.info("total: %6.2f%%", total * 100.)
