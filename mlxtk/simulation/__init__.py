import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .. import sge
from .base import SimulationBase


class Simulation(SimulationBase):
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

        parser = argparse.ArgumentParser(
            description="This is a mlxtk simulation")
        subparsers = parser.add_subparsers(dest="subcommand")

        subparsers.add_parser("graph")

        # parser for list
        subparsers.add_parser("list")

        # parser for run
        parser_run = subparsers.add_parser("run")
        parser_run.add_argument("-j",
                                "--jobs",
                                type=int,
                                default=1,
                                help="number of parallel workers")

        # parser for dry-run
        subparsers.add_parser("dry-run")

        # parser for clean
        parser_clean = subparsers.add_parser("clean")
        parser_clean.add_argument("-j",
                                  "--jobs",
                                  type=int,
                                  default=1,
                                  help="number of parallel workers")

        # parser for task-info
        parser_task_info = subparsers.add_parser("task-info")
        parser_task_info.add_argument("name")

        # parser for qsub
        parser_qsub = subparsers.add_parser("qsub")
        sge.add_parser_arguments(parser_qsub)

        # parser for qdel
        subparsers.add_parser("qdel")

        parser_propagation_status = subparsers.add_parser("propagation-status")
        parser_propagation_status.add_argument(
            "name",
            default="propagate",
            type=str,
            nargs="?",
            help="name of the propagation to check")

        args = parser.parse_args(argv)

        if args.subcommand is None:
            self.logger.error("No subcommand specified!")
            print()
            parser.print_help(sys.stderr)
            exit(1)

        if args.subcommand == "graph":
            self.cmd_graph(args)
        elif args.subcommand == "run":
            self.cmd_run(args)
        elif args.subcommand == "dry-run":
            self.cmd_dry_run(args)
        elif args.subcommand == "clean":
            self.cmd_clean(args)
        elif args.subcommand == "qsub":
            self.cmd_qsub(args)
        elif args.subcommand == "qdel":
            self.cmd_qdel(args)
        elif args.subcommand == "list":
            self.cmd_list(args)
        elif args.subcommand == "task-info":
            self.cmd_task_info(args)
        elif args.subcommand == "propagation-status":
            self.cmd_propagation_status(args)
