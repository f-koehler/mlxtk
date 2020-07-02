import argparse
import sys
from pathlib import Path
from typing import List, Optional

from mlxtk import sge
from mlxtk.simulation import base


class Simulation(base.SimulationBase):
    def __init__(self, name: Path = Path("sim"), working_dir: Optional[Path] = None):
        super().__init__(name, working_dir)

        self.argparser = argparse.ArgumentParser(description="This a mlxtk simulation")
        self.subparsers = self.argparser.add_subparsers()

        self.argparser_clean = self.subparsers.add_parser("clean")
        self.argparser_clean.set_defaults(subcommand=self.cmd_clean)
        self.argparser_clean.add_argument(
            "-j", "--jobs", type=int, default=1, help="number of parallel workers"
        )

        self.argparser_dry_run = self.subparsers.add_parser("dry-run")
        self.argparser_dry_run.set_defaults(subcommand=self.cmd_dry_run)

        self.argparser_graph = self.subparsers.add_parser("graph")
        self.argparser_graph.set_defaults(subcommand=self.cmd_graph)

        self.argparser_list = self.subparsers.add_parser("list")
        self.argparser_list.set_defaults(subcommand=self.cmd_list)

        self.argparser_propagation_status = self.subparsers.add_parser(
            "propagation-status"
        )
        self.argparser_propagation_status.set_defaults(
            subcommand=self.cmd_propagation_status
        )
        self.argparser_propagation_status.add_argument(
            "name",
            default="propagate",
            type=str,
            nargs="?",
            help="name of the propagation to check",
        )

        self.argparser_qdel = self.subparsers.add_parser("qdel")
        self.argparser_qdel.set_defaults(subcommand=self.cmd_qdel)

        self.argparser_qsub = self.subparsers.add_parser("qsub")
        self.argparser_qsub.set_defaults(subcommand=self.cmd_qsub)
        sge.add_parser_arguments(self.argparser_qsub)

        self.argparser_run = self.subparsers.add_parser("run")
        self.argparser_run.set_defaults(subcommand=self.cmd_run)
        self.argparser_run.add_argument(
            "-j", "--jobs", type=int, default=1, help="number of parallel workers"
        )

        self.argparser_task_info = self.subparsers.add_parser("task-info")
        self.argparser_task_info.set_defaults(subcommand=self.cmd_task_info)
        self.argparser_task_info.add_argument("name")

    from mlxtk.simulation.cmd_clean import cmd_clean
    from mlxtk.simulation.cmd_dry_run import cmd_dry_run
    from mlxtk.simulation.cmd_graph import cmd_graph
    from mlxtk.simulation.cmd_list import cmd_list
    from mlxtk.simulation.cmd_propagation_status import (
        cmd_propagation_status,
        check_propagation_status,
    )
    from mlxtk.simulation.cmd_qdel import cmd_qdel
    from mlxtk.simulation.cmd_qsub import cmd_qsub
    from mlxtk.simulation.cmd_run import cmd_run
    from mlxtk.simulation.cmd_task_info import cmd_task_info

    def main(self, argv: List[str] = None):
        if argv is None:
            argv = sys.argv[1:]

        args = self.argparser.parse_args(argv)
        args.subcommand(args)
