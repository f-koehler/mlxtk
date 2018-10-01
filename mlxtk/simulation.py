import argparse
import os
import subprocess
import sys

from . import cwd, doit_compat, log, sge


class Simulation(object):
    def __init__(self, name, working_dir=None):
        self.tasks = []
        self.working_dir = name if working_dir is None else working_dir
        self.logger = log.get_logger(__name__)

    def __iadd__(self, other):
        self.tasks += other
        return self

    def run(self, args):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        cwd.change_dir(self.working_dir)
        doit_compat.run_doit(self.tasks, ["-n", str(args.jobs)])
        cwd.go_back()

    def qsub(self, args):
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        cwd.change_dir(self.working_dir)
        sge.submit(
            [sys.executable, os.path.dirname(os.path.realpath(__file__)), "run"], args
        )
        cwd.go_back()

    def qdel(self, args):
        cwd.change_dir(self.working_dir)
        if os.path.exists("sge_stop"):
            LOGGER.warning("stopping job")
            subprocess.check_output("sge_stop")
        cwd.go_back()

    def task_info(self, args):
        cwd.change_dir(self.working_dir)
        doit_compat.run_doit(self.tasks, ["info", args.name])
        cwd.go_back()

    def main(self, args=sys.argv[1:]):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subcommand")

        subparsers.add_parser("list-tasks")

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
        elif args.subcommand == "run":
            self.run(args)
        elif args.subcommand == "qsub":
            self.qsub(args)
        elif args.subcommand == "qdel":
            self.qdel(args)
        elif args.subcommand == "list-tasks":
            doit_compat.run_doit(self.tasks, ["list"])
        elif args.subcommand == "task-info":
            self.task_info(args)
