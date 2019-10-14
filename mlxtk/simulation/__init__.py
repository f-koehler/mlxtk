import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .. import sge
from . import base


class Simulation(base.SimulationBase):
    def __init__(self, name: Path, working_dir: Optional[Path] = None):
        super().__init__(name, working_dir)

    from .cmd_clean import cmd_clean
    from .cmd_dry_run import cmd_dry_run
    from .cmd_graph import cmd_graph
    from .cmd_list import cmd_list
    from .cmd_propagation_status import cmd_propagation_status
    from .cmd_qdel import cmd_qdel
    from .cmd_qsub import cmd_qsub
    from .cmd_run import cmd_run
    from .cmd_task_info import cmd_task_info

    def main(self, argv: List[str] = None):
        if argv is None:
            argv = sys.argv[1:]

        args = self.argparser.parse_args(argv)
        args.subcommand(args)
