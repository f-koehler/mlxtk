import argparse
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from . import cwd, doit_compat, lock, log, sge
from .inout.output import read_output_ascii

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
                ["graph", "--backend=json", "--db-file=doit.json"],
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
                ["list", "--backend=json", "--db-file=doit.json"],
            )

    def run(self, args: argparse.Namespace):
        self.create_working_dir()
        with cwd.WorkingDir(self.working_dir):
            with lock.LockFile(Path("run.lock")):
                doit_compat.run_doit(
                    self.tasks_run,
                    [
                        "--process=" + str(args.jobs),
                        "--backend=json",
                        "--db-file=doit.json",
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
                        "--backend=json",
                        "--db-file=doit.json",
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
                        "--backend=json",
                        "--db-file=doit.json",
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
                    "--backend=json",
                    "--db-file=doit.json",
                    args.name,
                ],
            )

    def check_propagation_status(self, propagation: str) -> float:
        work_dir = self.working_dir / ("." + propagation)  # sim/.propagate/
        final_dir = self.working_dir / propagation  # sim/propagate/
        result_file = final_dir / "propagate.h5"  # sim/propagate/propagate.h5
        pickle_file = self.working_dir / (propagation + ".prop_pickle"
                                          )  # sim/propagate.prop_pickle
        output_file = work_dir / "output"  # sim/.propagate/output

        if not work_dir.exists():
            if result_file.exists():
                return 1.
            else:
                return 0.

        if not pickle_file.exists():
            return 0.

        if not output_file.exists():
            return 0.

        with open(pickle_file, "rb") as fptr:
            tfinal = pickle.load(fptr)[3]["tfinal"]

        times, _, _, _ = read_output_ascii(output_file)

        return times[-1] / tfinal

    def propagation_status(self, args: argparse.Namespace):
        self.logger.info("check progress of propagation: %s", args.name)
        progress = self.check_propagation_status(args.name)
        self.logger.info("total progress: %6.2f%%", progress * 100.)

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
        elif args.subcommand == "propagation-status":
            self.propagation_status(args)
