import argparse
import os
import subprocess
import sys
from typing import Iterable, Optional

from . import cwd, doit_compat, log, sge

LOGGER = log.get_logger(__name__)


class Simulation:
    def __init__(self, name: str, working_dir: Optional[str] = None):
        self.name = name
        self.working_dir = name if working_dir is None else working_dir
        self.task_generators = []
        self.logger = log.get_logger(__name__)

    def __iadd__(self, generators):
        self.task_generators += generators
        return self

    def create_working_dir(self):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

    def run(self, args: argparse.Namespace):
        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir):
            doit_compat.run_doit(self.task_generators, ["-n", str(args.jobs)])

    def qsub(self, args: argparse.Namespace):
        self.create_working_dir()
        sge.submit(
            " ".join([sys.executable, os.path.abspath(sys.argv[0]), "run"]),
            args,
            sge_dir=self.working_dir,
        )

    def qdel(self, args: argparse.Namespace):
        del args

        if not os.path.exists(self.working_dir):
            self.logger.warning(
                "working dir %s does not exist, do nothing", self.working_dir
            )
            return
        with cwd.WorkingDir(self.working_dir):
            if os.path.exists("sge_stop"):
                LOGGER.warning("stopping job")
                subprocess.check_output("sge_stop")

    def task_info(self, args: argparse.Namespace):
        with cwd.WorkingDir(self.working_dir):
            doit_compat.run_doit(self.task_generators, ["info", args.name])

    def main(self, args: Iterable[str] = sys.argv[1:]):
        parser = argparse.ArgumentParser(description="This is a mlxtk simulation")
        subparsers = parser.add_subparsers(dest="subcommand")

        subparsers.add_parser("list")

        # parser for run
        parser_run = subparsers.add_parser("run")
        parser_run.add_argument(
            "-j", "--jobs", type=int, default=1, help="number of parallel workers"
        )

        # parser for task-info
        parser_task_info = subparsers.add_parser("task-info")
        parser_task_info.add_argument("name")

        # parser for qsub
        parser_qsub = subparsers.add_parser("qsub")
        sge.add_parser_arguments(parser_qsub)

        # parser for qdel
        subparsers.add_parser("qdel")

        args = parser.parse_args(args)

        if args.subcommand is None:
            self.logger.error("No subcommand specified!")
            print()
            parser.print_help(sys.stderr)
            exit(1)

        if args.subcommand == "run":
            self.run(args)
        elif args.subcommand == "qsub":
            self.qsub(args)
        elif args.subcommand == "qdel":
            self.qdel(args)
        elif args.subcommand == "list":
            doit_compat.run_doit(self.task_generators, ["list"])
        elif args.subcommand == "task-info":
            self.task_info(args)
