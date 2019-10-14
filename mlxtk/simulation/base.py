import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional

from .. import sge
from ..inout.output import read_output_ascii
from ..log import get_logger


class SimulationBase:
    def __init__(self, name: Path, working_dir: Optional[Path] = None):
        self.name = name
        self.working_dir = (Path(name)
                            if working_dir is None else working_dir).resolve()
        self.tasks_run = []
        self.tasks_clean = []
        self.tasks_dry_run = []
        self.logger = get_logger(__name__ + ".Simulation")

        self.argparser = argparse.ArgumentParser(
            description="This a mlxtk simulation")
        self.subparsers = self.argparser.add_subparsers()

        self.argparser_clean = self.subparsers.add_parser("clean")
        self.argparser_clean.set_defaults(subcommand=self.cmd_clean)
        self.argparser_clean.add_argument("-j",
                                          "--jobs",
                                          type=int,
                                          default=1,
                                          help="number of parallel workers")

        self.argparser_dry_run = self.subparsers.add_parser("dry-run")
        self.argparser_dry_run.set_defaults(subcommand=self.cmd_dry_run)

        self.argparser_graph = self.subparsers.add_parser("graph")
        self.argparser_graph.set_defaults(subcommand=self.cmd_graph)

        self.argparser_list = self.subparsers.add_parser("list")
        self.argparser_list.set_defaults(subcommand=self.cmd_list)

        self.argparser_propagation_status = self.subparsers.add_parser(
            "propagation-status")
        self.argparser_propagation_status.set_defaults(
            subcommand=self.cmd_propagation_status)
        self.argparser_propagation_status.add_argument(
            "name",
            default="propagate",
            type=str,
            nargs="?",
            help="name of the propagation to check")

        self.argparser_qdel = self.subparsers.add_parser("qdel")
        self.argparser_qdel.set_defaults(subcommand=self.cmd_qdel)

        self.argparser_qsub = self.subparsers.add_parser("qsub")
        self.argparser_qsub.set_defaults(subcommand=self.cmd_qsub)
        sge.add_parser_arguments(self.argparser_qsub)

        self.argparser_run = self.subparsers.add_parser("run")
        self.argparser_run.set_defaults(subcommand=self.cmd_run)
        self.argparser_run.add_argument("-j",
                                        "--jobs",
                                        type=int,
                                        default=1,
                                        help="number of parallel workers")

        self.argparser_task_info = self.subparsers.add_parser("task-info")
        self.argparser_task_info.set_defaults(subcommand=self.cmd_task_info)
        self.argparser_task_info.add_argument("name")

    def __iadd__(self, generator):
        self.tasks_run += generator()
        self.tasks_clean += generator.get_tasks_clean()
        self.tasks_dry_run += generator.get_tasks_dry_run()
        return self

    def create_working_dir(self):
        if not self.working_dir.exists():
            os.makedirs(self.working_dir)
