import argparse

from ..cwd import WorkingDir
from ..util import compress_folder
from .base import SimulationSetBase


def cmd_archive(self: SimulationSetBase, args: argparse.Namespace):
    self.create_working_dir()
    with WorkingDir(self.working_dir.resolve().parent):
        compress_folder(self.working_dir.name,
                        compression=args.compression,
                        jobs=args.jobs)
