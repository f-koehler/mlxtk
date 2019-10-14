"""Work with a collection of simulations.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Union

from ..log import get_logger
from ..simulation import Simulation
from ..util import make_path


class SimulationSetBase:
    """A collection of simulations.

    Args:
        name (str): name for this set of simulations
        simulations (list): the list of simulations
        working_dir (str): optional working directory

    Attributes:
        name (str): name for this set of simulations
        simulations (list): the list of simulations
        working_dir (str): optional working directory (defaults to ``name``)
    """
    def __init__(
            self,
            name: str,
            simulations: List[Simulation],
            working_dir: Union[str, Path] = None,
    ):
        self.name = name
        self.simulations = simulations
        self.working_dir = Path(
            name).resolve() if working_dir is None else make_path(working_dir)

        self.logger = get_logger(__name__ + ".SimulationSet")

        self.argparser = argparse.ArgumentParser(
            description="This is a set of mlxtk simulations")
        self.construct_argparser()

    from .cli import construct_argparser
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

    def create_working_dir(self):
        if not self.working_dir.exists():
            os.makedirs(self.working_dir)

    def main(self, argv: List[str]):
        if argv is None:
            argv = sys.argv[1:]

        args = self.argparser.parse_args(argv)

        if args.subcommand is None:
            self.logger.error("No subcommand specified!")
            print()
            self.argparser.print_help(sys.stderr)
            exit(1)

        subcommand_map = {
            "archive": self.cmd_archive,
            "clean": self.cmd_clean,
            "dry-run": self.cmd_dry_run,
            "list": self.cmd_list,
            "list-tasks": self.cmd_list_tasks,
            "lockfiles": self.cmd_lockfiles,
            "qdel": self.cmd_qdel,
            "qsub": self.cmd_qsub_array,
            "run": self.cmd_run,
            "run-index": self.cmd_run_index,
            "task-info": self.cmd_task_info,
            "propagation-status": self.cmd_propagation_status
        }

        subcommand_map[args.subcommand](args)
