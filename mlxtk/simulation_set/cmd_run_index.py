import argparse
import sys
from pathlib import Path

from mlxtk.cwd import WorkingDir
from mlxtk.simulation_set.base import SimulationSetBase


def cmd_run_index(self: SimulationSetBase, args: argparse.Namespace):
    script_dir = Path(sys.argv[0]).parent.resolve()
    with WorkingDir(script_dir):
        self.logger.info("run simulation with index %d", args.index)
        self.simulations[args.index].main(["run"])
