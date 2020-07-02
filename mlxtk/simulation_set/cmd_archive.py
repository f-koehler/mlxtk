import argparse

from mlxtk.cwd import WorkingDir
from mlxtk.simulation_set.base import SimulationSetBase
from mlxtk.util import compress_folder


def cmd_archive(self: SimulationSetBase, args: argparse.Namespace):
    self.create_working_dir()
    with WorkingDir(self.working_dir.resolve().parent):
        compress_folder(
            self.working_dir.name, compression=args.compression, jobs=args.jobs
        )
