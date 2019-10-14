import sys
from pathlib import Path
from typing import List, Union

from ..simulation import Simulation
from . import base


class SimulationSet(base.SimulationSetBase):
    def __init__(
            self,
            name: str,
            simulations: List[Simulation],
            working_dir: Union[str, Path] = None,
    ):
        super().__init__(name, simulations, working_dir)

    from .cmd_archive import cmd_archive
    from .cmd_clean import cmd_clean
    from .cmd_dry_run import cmd_dry_run
    from .cmd_list_tasks import cmd_list_tasks
    from .cmd_list import cmd_list
    from .cmd_lockfiles import cmd_lockfiles
    from .cmd_propagation_status import cmd_propagation_status
    from .cmd_qdel import cmd_qdel
    from .cmd_qsub_array import cmd_qsub_array
    from .cmd_run_index import cmd_run_index
    from .cmd_run import cmd_run
    from .cmd_task_info import cmd_task_info

    def main(self, argv: List[str]):
        if argv is None:
            argv = sys.argv[1:]

        args = self.argparser.parse_args(argv)
        args.subcommand(args)
