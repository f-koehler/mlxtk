import argparse
import sys
from pathlib import Path

from mlxtk import sge
from mlxtk.cwd import WorkingDir
from mlxtk.simulation.base import SimulationBase


def cmd_qsub(self: SimulationBase, args: argparse.Namespace):
    self.create_working_dir()
    call_dir = Path.cwd().absolute()
    script_path = Path(sys.argv[0]).absolute()
    with WorkingDir(self.working_dir):
        sge.submit(
            " ".join([sys.executable, str(script_path), "run"]),
            args,
            sge_dir=call_dir,
            job_name=self.name,
        )
