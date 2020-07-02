import argparse
import subprocess
from pathlib import Path

from mlxtk.cwd import WorkingDir
from mlxtk.simulation.base import SimulationBase


def cmd_qdel(self: SimulationBase, args: argparse.Namespace):
    del args

    if not self.working_dir.exists():
        self.logger.warning(
            "working dir %s does not exist, do nothing", self.working_dir
        )
        return
    with WorkingDir(self.working_dir):
        if Path("sge_stop").exists():
            self.logger.warning("stopping job")
            subprocess.check_output("sge_stop")
