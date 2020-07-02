import argparse

from mlxtk.cwd import WorkingDir
from mlxtk.simulation_set.base import SimulationSetBase


def cmd_qdel(self: SimulationSetBase, args: argparse.Namespace):
    del args

    if not self.working_dir.exists():
        self.logger.warning(
            "working dir %s does not exist, do nothing", self.working_dir
        )

    with WorkingDir(self.working_dir):
        for simulation in self.simulations:
            simulation.main(["qdel"])
