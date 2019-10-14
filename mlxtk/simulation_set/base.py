"""Work with a collection of simulations.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Union

from .. import sge
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

    def create_working_dir(self):
        if not self.working_dir.exists():
            os.makedirs(self.working_dir)
