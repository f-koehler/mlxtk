import argparse
import os
import sys
from typing import Iterable, List, Optional

from .log import get_logger
from . import cwd
from .simulation import Simulation


class SimulationSet(object):
    def __init__(
        self,
        name: str,
        simulations: List[Simulation],
        working_dir: Optional[str] = None,
    ):
        self.name = name
        self.simulations = simulations
        self.working_dir = name if working_dir is None else working_dir
        self.logger = get_logger(__name__)

        self.argparser = argparse.ArgumentParser(
            description="This is a set of mlxtk simulations"
        )
        subparsers = self.argparser.add_subparsers(dest="subcommand")

        subparsers.add_parser("list")
        self.argparser_list_tasks = subparsers.add_parser("list-tasks")
        subparsers.add_parser("qdel")
        subparsers.add_parser("qsub")
        subparsers.add_parser("run")

        self.argparser_list_tasks.add_argument(
            "index", type=int, help="index of the simulation whose tasks to list"
        )

    def create_working_dir(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    def list_(self, args: argparse.Namespace):
        for i, simulation in enumerate(self.simulations):
            print(i, simulation.name)

    def list_tasks(self, args: argparse.Namespace):
        self.simulations[args.index].main(["list"])

    def run(self, args: argparse.Namespace):
        self.create_working_dir()

        cwd.change_dir(self.working_dir)
        for simulation in self.simulations:
            simulation.main(["run"])
        cwd.go_back()

    def qdel(self, args: argparse.Namespace):
        if not os.path.exists(self.working_dir):
            self.logger.warning(
                "working dir %s does not exist, do nothing", self.working_dir
            )

        cwd.change_dir(self.working_dir)
        for simulation in self.simulations:
            simulation.main(["qdel"])
        cwd.go_back()

    def qsub(self, args: argparse.Namespace):
        self.create_working_dir()

        cwd.change_dir(self.working_dir)
        for simulation in self.simulations:
            simulation.qsub(args)
        cwd.go_back()

    def main(self, args: Iterable[str] = sys.argv[1:]):

        args = self.argparser.parse_args(args)

        if args.subcommand is None:
            self.logger.error("No subcommand specified!")
            print()
            self.argparser.print_help(sys.stderr)
            exit(1)

        if args.subcommand == "list":
            self.list_(args)
        elif args.subcommand == "qdel":
            self.qdel(args)
        elif args.subcommand == "qsub":
            self.qsub(args)
        elif args.subcommand == "run":
            self.run(args)
