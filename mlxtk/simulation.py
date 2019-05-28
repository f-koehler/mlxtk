import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import cwd, doit_compat, lock, log, sge

LOGGER = log.get_logger(__name__)


class Simulation:
    def __init__(self, name: Path, working_dir: Optional[Path] = None):
        self.name = name
        self.working_dir = (Path(name)
                            if working_dir is None else working_dir).resolve()
        self.tasks_run = []
        self.tasks_clean = []
        self.tasks_dry_run = []
        self.logger = log.get_logger(__name__ + ".Simulation")

    def __iadd__(self, generator):
        self.tasks_run += generator()
        self.tasks_clean += generator.get_tasks_clean()
        self.tasks_dry_run += generator.get_tasks_dry_run()
        return self

    def create_working_dir(self):
        if not self.working_dir.exists():
            os.makedirs(self.working_dir)

    def graph(self, args: argparse.Namespace):
        del args

        regex = re.compile(r"^\s+\".+\"\s*->\s*\".+\";$")

        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir):
            doit_compat.run_doit(
                self.tasks_run,
                ["graph", "--backend=sqlite3", "--db-file=doit.sqlite3"],
            )

            with open("tasks.dot") as fptr:
                code = fptr.readlines()

            for i, _ in enumerate(code):
                if not regex.match(code[i]):
                    continue

                code[i] = code[i].replace(":", "\\n")

            with open("tasks.dot", "w") as fptr:
                fptr.writelines(code)

    def list(self, args: argparse.Namespace):
        del args

        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir):
            doit_compat.run_doit(
                self.tasks_run,
                ["list", "--backend=sqlite3", "--db-file=doit.sqlite3"],
            )

    def run(self, args: argparse.Namespace):
        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir):
            with lock.LockFile(Path("run.lock")):
                doit_compat.run_doit(
                    self.tasks_run,
                    [
                        "--process=" + str(args.jobs),
                        "--backend=sqlite3",
                        "--db-file=doit.sqlite3",
                    ],
                )

    def dry_run(self, args: argparse.Namespace):
        del args

        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir):
            with lock.LockFile(Path("run.lock")):
                doit_compat.run_doit(
                    self.tasks_dry_run,
                    [
                        "--backend=sqlite3",
                        "--db-file=doit.sqlite3",
                    ],
                )

    def clean(self, args: argparse.Namespace):
        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir):
            with lock.LockFile(Path("run.lock")):
                doit_compat.run_doit(
                    self.tasks_clean,
                    [
                        "--process=" + str(args.jobs),
                        "--backend=sqlite3",
                        "--db-file=doit.sqlite3",
                    ],
                )

    def qsub(self, args: argparse.Namespace):
        self.create_working_dir()
        call_dir = Path.cwd().absolute()
        script_path = Path(sys.argv[0]).absolute()
        with cwd.WorkingDir(self.working_dir):
            sge.submit(" ".join([sys.executable,
                                 str(script_path), "run"]),
                       args,
                       sge_dir=call_dir,
                       job_name=self.name)

    def qdel(self, args: argparse.Namespace):
        del args

        if not self.working_dir.exists():
            self.logger.warning("working dir %s does not exist, do nothing",
                                self.working_dir)
            return
        with cwd.WorkingDir(self.working_dir):
            if Path("sge_stop").exists():
                self.logger.warning("stopping job")
                subprocess.check_output("sge_stop")

    def task_info(self, args: argparse.Namespace):
        with cwd.WorkingDir(self.working_dir):
            doit_compat.run_doit(
                self.tasks_run,
                [
                    "info",
                    "--backend=sqlite3",
                    "--db-file=doit.sqlite3",
                    args.name,
                ],
            )

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

        args = parser.parse_args(argv)

        if args.subcommand is None:
            self.logger.error("No subcommand specified!")
            print()
            parser.print_help(sys.stderr)
            exit(1)

        if args.subcommand == "graph":
            self.graph(args)
        elif args.subcommand == "run":
            self.run(args)
        elif args.subcommand == "dry-run":
            self.dry_run(args)
        elif args.subcommand == "clean":
            self.clean(args)
        elif args.subcommand == "qsub":
            self.qsub(args)
        elif args.subcommand == "qdel":
            self.qdel(args)
        elif args.subcommand == "list":
            self.list(args)
        elif args.subcommand == "task-info":
            self.task_info(args)
