import argparse
import sys
from pathlib import Path
from typing import List, Union

from .. import sge
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

        self.argparser = argparse.ArgumentParser(
            description="This is a set of mlxtk simulations")
        self.subparsers = self.argparser.add_subparsers()

        self.argparser_archive = self.subparsers.add_parser("archive")
        self.argparser_archive.set_defaults(subcommand=self.cmd_archive)
        self.argparser_archive.add_argument(
            "-c",
            "--compression",
            type=int,
            default=9,
            help="compression level [1-9] (1: fastest, 9: best)")
        self.argparser_archive.add_argument(
            "-j",
            "--jobs",
            type=int,
            default=1,
            help="number of jobs (when pigz is available)")

        self.argparser_clean = self.subparsers.add_parser("clean")
        self.argparser_clean.set_defaults(subcommand=self.cmd_clean)
        self.argparser_clean.add_argument("-j",
                                          "--jobs",
                                          type=int,
                                          default=1,
                                          help="number of parallel workers")

        self.argparser_dry_run = self.subparsers.add_parser("dry-run")
        self.argparser_dry_run.set_defaults(subcommand=self.cmd_dry_run)

        self.argparser_list = self.subparsers.add_parser("list")
        self.argparser_list.set_defaults(subcommand=self.cmd_list)
        self.argparser_list.add_argument("-d",
                                         "--directory",
                                         action="store_true")

        self.argparser_list_tasks = self.subparsers.add_parser("list-tasks")
        self.argparser_list_tasks.set_defaults(subcommand=self.cmd_list_tasks)
        self.argparser_list_tasks.add_argument(
            "index",
            type=int,
            help="index of the simulation whose tasks to list")

        self.argparser_lockfiles = self.subparsers.add_parser("lockfiles")
        self.argparser_lockfiles.set_defaults(subcommand=self.cmd_lockfiles)

        self.argparser_propagation_status = self.subparsers.add_parser(
            "propagation-status")
        self.argparser_propagation_status.set_defaults(
            subcommand=self.cmd_propagation_status)
        self.argparser_propagation_status.add_argument(
            "name",
            default="propagate",
            nargs="?",
            type=str,
            help="name of the propagation")

        self.argparser_qsub = self.subparsers.add_parser("qsub")
        self.argparser_qsub.set_defaults(subcommand=self.cmd_qsub_array)
        sge.add_parser_arguments(self.argparser_qsub)

        self.argparser_run = self.subparsers.add_parser("run")
        self.argparser_run.set_defaults(subcommand=self.cmd_run)
        self.argparser_run.add_argument("-j",
                                        "--jobs",
                                        type=int,
                                        default=1,
                                        help="number of parallel workers")

        self.argparser_run_index = self.subparsers.add_parser("run-index")
        self.argparser_run_index.set_defaults(subcommand=self.cmd_run_index)
        self.argparser_run_index.add_argument(
            "index", type=int, help="index of the simulation to run")

        self.argparser_task_info = self.subparsers.add_parser("task-info")
        self.argparser_task_info.set_defaults(subcommand=self.cmd_task_info)
        self.argparser_task_info.add_argument("index",
                                              type=int,
                                              help="index of the simulation")
        self.argparser_task_info.add_argument("name",
                                              type=str,
                                              help="name of the task")

        self.argparser_qdel = self.subparsers.add_parser("qdel")
        self.argparser_qdel.set_defaults(subcommand=self.cmd_qdel)

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
