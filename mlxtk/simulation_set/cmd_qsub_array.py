import argparse
import os
import sys
from pathlib import Path

from mlxtk import sge
from mlxtk.cwd import WorkingDir
from mlxtk.simulation_set.base import SimulationSetBase


def cmd_qsub(self: SimulationSetBase, args: argparse.Namespace):
    self.logger.info("submitting simulation set to SGE scheduler")
    self.create_working_dir()

    set_dir = self.working_dir.resolve()
    script_path = Path(sys.argv[0]).resolve()

    for index, simulation in enumerate(self.simulations):
        simulation_dir = set_dir / simulation.working_dir
        if not simulation_dir.exists():
            os.makedirs(simulation_dir)
        with WorkingDir(simulation_dir):
            sge.submit(
                " ".join([sys.executable, str(script_path), "run-index", str(index)]),
                args,
                sge_dir=script_path.parent,
                job_name=simulation.name,
            )


def cmd_qsub_array(self: SimulationSetBase, args: argparse.Namespace):
    self.logger.info("submitting simulation set as an array to SGE scheduler")
    self.create_working_dir()

    script_path = Path(sys.argv[0]).resolve()
    command = " ".join([sys.executable, str(script_path), "run-index"])

    with WorkingDir(self.working_dir):
        sge.submit_array(
            command,
            len(self.simulations),
            args,
            sge_dir=script_path.parent,
            job_name=self.name,
        )
