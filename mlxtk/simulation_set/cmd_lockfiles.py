import argparse
import json

from mlxtk.simulation_set.base import SimulationSetBase


def cmd_lockfiles(self: SimulationSetBase, args: argparse.Namespace):
    lock_files = [
        self.working_dir.resolve() / sim.working_dir / "run.lock"
        for sim in self.simulations
    ]
    working_dirs = [sim.working_dir for sim in self.simulations]

    counter = 0
    for lock_file, working_dir in zip(lock_files, working_dirs):
        if not lock_file.exists():
            continue

        counter += 1

        with open(lock_file, "r") as fptr:
            self.logger.info("lock: %s", working_dir)
            lock = json.load(fptr)
            try:
                self.logger.info("\thost: %s", lock["host"])
                self.logger.info("\tpid:  %d", lock["pid"])
            except KeyError:
                pass

    self.logger.info("%d lock files", counter)
